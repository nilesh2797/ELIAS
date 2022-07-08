#!/usr/bin/env python
# coding: utf-8

from accelerate import Accelerator
from accelerate.utils import convert_outputs_to_fp32
accelerator = Accelerator()
def PLOG(msg, attr='yellow'):
    accelerator.print(_c(f'\n<<{"-"*(50-(len(msg)+1)//2)} {msg} {"-"*(50-len(msg)//2)}>>', attr=attr), flush=True)

from resources import _c, read_sparse_mat, get_text, _filter, XCMetrics
from dl_base import ToD, PreTokBertDataset, FixedDataset, XCCollator, TransformerInputLayer
from nets import *
from losses import *

import json, sys, os, socket
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import xclib.data.data_utils as data_utils
import xclib.evaluation.xc_metrics as xc_metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

# Cluster gen and sparse ranker utils
import pecos
from pecos.utils import smat_util
from pecos.xmc.xlinear.model import XLinearModel
from pecos.xmc import Indexer, LabelEmbeddingFactory, MLModel, MLProblem

transformers.set_seed(42)

PLOG(f'ELIAS {" ".join(sys.argv[1:])}', 'green')
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--project', default='ELIAS')
parser.add_argument('--expname', default='')
parser.add_argument('--dataset', required=True)
parser.add_argument('--maxlen', type=int, default=32, help='max seq length for transformer')
parser.add_argument('--net', default='ELIAS')
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--loss', default='joint')
parser.add_argument('--loss-with-logits', action='store_true', default=False)
parser.add_argument('--joint-loss-gamma', type=float, default=0.05)
parser.add_argument('--tf', default='bert-base-uncased')
parser.add_argument('--tf-pooler', default='cls')

parser.add_argument('--C', type=int, default=0)
parser.add_argument('--max-leaf', type=int, default=100)
parser.add_argument('--beam', type=int, default=20)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta-gain', type=float, default=1.5)
parser.add_argument('--K', type=int, default=2000)
parser.add_argument('--kappa', type=int, default=1000)
# parser.add_argument('--beam-loss-gamma', type=float, default=0)

parser.add_argument('--warmup', type=float, default=0.1, help='warmup steps percentage')
parser.add_argument('--W-accum-steps', type=int, default=1)
parser.add_argument('--no-amp-encode', dest='amp_encode', action='store_false', help='do not use amp for encoder')
parser.add_argument('--clf-dim', type=int, default=0)
parser.add_argument('--bottleneck', type=int, default=0)
parser.add_argument('--norm-embs', action='store_true', help='Normalize encoder embeddings', default=False)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--xc-lr', type=float, default=0.01)
parser.add_argument('--dense-lr', type=float, default=-1.0)
parser.add_argument('--enc-lr', type=float, default=5e-5)
parser.add_argument('--bsz', type=int, default=128)
parser.add_argument('--num-epochs', type=int, default=25)
parser.add_argument('--eval-interval', type=int, default=3)
parser.add_argument('--use-swa', dest='use_swa', action='store_true')
parser.add_argument('--swa-step', type=int, default=1000)
parser.add_argument('--swa-start', type=int, default=8)
parser.add_argument('--no-cmat-update', dest='cmat_update', action='store_false', help='do not update cluster in stage1')

parser.add_argument('--embs-dir', default='')
parser.add_argument('--save', action='store_true', help='save model and score mat', default=False)
parser.add_argument('--online-tok', action='store_true', help='use online tokenized dataset', default=False)
parser.add_argument('--resume-path', help='continue training from previous checkpoint', default='')
parser.add_argument('--cmat-file', help='saved cluster matrix npz file', default='')
parser.add_argument('--cmat-seed', help='initial cluster matrix seed', default=-1, type=int)
parser.add_argument('--A-init-file', help='saved candidates npz file', default='')
parser.add_argument('--save-embs', action='store_true', default=False)
parser.add_argument('--gen-A-init', action='store_true', help='generate an initialization for A with final model', default=False)
parser.add_argument('--ranker', action='store_true', default=False)
parser.add_argument('--ranker-alpha', type=float, default=0.5)
parser.add_argument('--ranker-topk', type=int, default=100)
parser.add_argument('--score-calibrate', action='store_true', default=False)
parser.set_defaults(amp_encode=True)
parser.set_defaults(use_swa=False)
parser.set_defaults(cmat_update=True)

args = parser.parse_args()
if args.expname != '' and args.num_epochs == 0:
    args.__dict__ = json.load(open(f'Results/{args.project}/{args.dataset}/{args.expname}/args.json'))
    parser.set_defaults(**args.__dict__)
    args = parser.parse_args()
    
args.device = str(accelerator.device)
args.amp = accelerator.state.use_fp16
args.num_gpu = accelerator.state.num_processes
args.use_grad_scaler = ((not args.amp) and (args.num_gpu == 1) and args.amp_encode)
args.hostname = socket.gethostname()

cmat_seed_suffix = '' if args.cmat_seed < 0 == '' else f'-{args.cmat_seed}'
args.netname = args.net if args.net == 'fcnet' else f'{args.net}-{args.stage}-leaf{args.max_leaf}-beam{args.beam}{cmat_seed_suffix}'
if args.expname == '':
    args.expname = f"{args.netname}-py_{args.loss}_tf-{args.tf}{'-norm' if args.norm_embs else ''}{'_'+args.embs_dir if args.tf.lower() == 'none' else ''}_xc-lr-{args.xc_lr}_enc-lr-{args.enc_lr}_bsz-{args.bsz}"
args.dense_lr = args.xc_lr if args.dense_lr < 0 else args.dense_lr
args.tf = None if args.tf.lower() == 'none' else args.tf
args.embs_dim = 768

args.DATA_DIR = DATA_DIR = f'Datasets/{args.dataset}'
args.OUT_DIR = OUT_DIR = f'Results/{args.project}/{args.dataset}/{args.expname}'
os.makedirs(OUT_DIR, exist_ok=True)
accelerator.print(f'expname: {_c(args.expname, attr="blue")}, dataset: {_c(args.dataset, attr="blue")}')

PLOG('Loading Dataset')
trn_X_Y = sp.load_npz(f'{DATA_DIR}/Y.trn.npz')
tst_X_Y = sp.load_npz(f'{DATA_DIR}/Y.tst.npz')

args.numy = trn_X_Y.shape[1]
args.C = int(2**np.ceil(np.log2(args.numy/args.max_leaf)))

if not os.path.exists(args.A_init_file):
    if args.stage == 1:
        assert args.cmat_update
        args.A_init_file = f'{args.OUT_DIR}/cmat.npz'
    else:
        assert os.path.exists(f'{os.path.dirname(args.resume_path)}/A_init.npz')
        args.A_init_file = f'{os.path.dirname(args.resume_path)}/A_init.npz'
    accelerator.print(f'Setting args.A_init_file file to: {args.A_init_file}')
    
if "amazon" in args.dataset.lower(): A = 0.6; B = 2.6
elif "wiki" in args.dataset.lower() and "wikiseealso" not in args.dataset.lower(): A = 0.5; B = 0.4
else : A = 0.55; B = 1.5
inv_prop = xc_metrics.compute_inv_propesity(trn_X_Y, A, B)
trn_filter_mat = tst_filter_mat = None
if os.path.exists('%s/filter_labels_test.txt'%(DATA_DIR)):
    temp = np.fromfile('%s/filter_labels_test.txt'%(DATA_DIR), sep=' ').astype(int)
    temp = temp.reshape(-1, 2).T
    tst_filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), tst_X_Y.shape).tocsr()

if args.tf is not None:
    args.token_type = 'bert-base-uncased'
    if 'roberta' in args.tf: args.token_type = 'roberta-base'
    if not args.online_tok:
        trn_dataset = PreTokBertDataset(f'{DATA_DIR}/{args.token_type}-{args.maxlen}', trn_X_Y, args.maxlen, args.token_type, doc_type='trn')
        tst_dataset = PreTokBertDataset(f'{DATA_DIR}/{args.token_type}-{args.maxlen}', tst_X_Y, args.maxlen, args.token_type, doc_type='tst')
    else:
        trnX = [x.strip() for x in open(f'{DATA_DIR}/raw/trn_X.txt').readlines()]
        tstX = [x.strip() for x in open(f'{DATA_DIR}/raw/tst_X.txt').readlines()]
        trn_dataset = OnlineBertDataset(trnX, trn_X_Y, args.maxlen, args.token_type)
        tst_dataset = OnlineBertDataset(tstX, tst_X_Y, args.maxlen, args.token_type)
else:
    args.embs_dir = f'Datasets/{args.dataset}/embs/{args.embs_dir}'
    trn_embs = np.load(f'{args.embs_dir}/trn_embs.npy')
    tst_embs = np.load(f'{args.embs_dir}/tst_embs.npy')
    args.embs_dim = trn_embs.shape[1]
    trn_dataset = FixedDataset(trn_embs, trn_X_Y)
    tst_dataset = FixedDataset(tst_embs, tst_X_Y)

trn_loader = torch.utils.data.DataLoader(
    trn_dataset,
    batch_size=args.bsz,
    num_workers=4,
    collate_fn=XCCollator(trn_X_Y.shape[1], trn_dataset),
    shuffle=True,
    pin_memory=True)

tst_loader = torch.utils.data.DataLoader(
    tst_dataset,
    batch_size=args.bsz*2,
    num_workers=4,
    collate_fn=XCCollator(trn_X_Y.shape[1], tst_dataset),
    shuffle=False,
    pin_memory=True)

PLOG('Init Training')
if accelerator.is_main_process and args.num_epochs > 0:
    with open(f'{OUT_DIR}/args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    import wandb
    wandb.init(project=f'{args.project}_{args.dataset}', name=args.expname)
    wandb.config.update(args)

accelerator.wait_for_everyone()
net = NETS[args.net](args)
criterion = LOSSES[args.loss](args)

if args.resume_path != '':
    accelerator.print(f'Loading net state dict from: {args.resume_path}')
    loaded_state = torch.load(args.resume_path, map_location='cpu')
    loaded_state = {k: v for k, v in loaded_state.items() if not (k.startswith('transform') or k.startswith('A_nz_vals') or k.startswith('w_leaf'))}
    accelerator.print(net.load_state_dict(loaded_state, strict=False))

def _no_decay(n,p):
    if n.startswith('encoder'):
        return any([x in n.lower() for x in ['bias', 'layernorm', 'layer_norm']])
    return True

def _is_enc_param(n,p):
    return n.startswith('encoder')

def _is_xc_param(n,p):
    return any((x == args.numy or x == (args.numy+1)) for x in p.shape) or n == 'A_nz_vals'

def _is_dense_param(n,p):
    return (not _is_enc_param(n,p)) and (not _is_xc_param(n,p))

optim_wrap = {
    'enc': {'class': torch.optim.AdamW, 'params': [], 'f': _is_enc_param, 'accum_steps': 1, 'args': {'lr': args.enc_lr}},
    'xc' : {'class': torch.optim.AdamW, 'params': [], 'f': _is_xc_param, 'accum_steps': args.W_accum_steps, 'args': {'lr': args.xc_lr}},
    'dense': {'class': torch.optim.AdamW, 'params': [], 'f': _is_dense_param, 'accum_steps': args.W_accum_steps, 'args': {'lr': args.dense_lr}}
    }

for n,p in net.named_parameters():
    for k in optim_wrap.keys():
        if p.requires_grad and optim_wrap[k]['f'](n,p):
            optim_wrap[k]['params'].append((n, p))

optims = {}
for k, v in optim_wrap.items():
    grouped_params = [{'params': [p for n,p in v['params'] if _no_decay(n,p)], 'weight_decay': 0}, 
                      {'params': [p for n,p in v['params'] if not _no_decay(n,p)], 'weight_decay': 0.01}]
    if len(v['params']) > 0: optims[k] = v['class'](grouped_params, **v['args'])

accelerator.print(net)
for k in optim_wrap.keys():
    accelerator.print(_c(f'{k} parameters ({optim_wrap[k]["args"]}, accum: {optim_wrap[k]["accum_steps"]}): ', attr='bold blue'), [p[0] for p in optim_wrap[k]['params']])

net, optims, trn_loader = accelerator.prepare(net, optims, trn_loader)
if isinstance(net, torch.nn.parallel.DistributedDataParallel):
    net.ToD = net.module.ToD
    net.encode = net.module.encode
    net.parent = net.module.parent
    net.update = net.module.update
    net.predict = net.module.predict
    net.save_model = net.module.save_model
    net.swa_init = net.module.swa_init
    net.swa_swap_params = net.module.swa_swap_params
    net.swa_step = net.module.swa_step

total_steps = {k: len(trn_loader)*args.num_epochs/optim_wrap[k]['accum_steps'] for k in optims.keys()}
schedulers = {k: transformers.get_linear_schedule_with_warmup(optim, num_warmup_steps=int(args.warmup*total_steps[k]), num_training_steps=total_steps[k]) for k, optim in optims.items()}
# schedulers = {k: transformers.get_constant_schedule_with_warmup(optim, num_warmup_steps=int(args.warmup*total_steps[k])) for k, optim in optims.items()}

PLOG('Training Loop')
best_ndcg = -1
scaler = torch.cuda.amp.GradScaler()
# net.forward = convert_outputs_to_fp32(torch.cuda.amp.autocast(enabled=args.use_grad_scaler)(net.forward))

for epoch in range(args.num_epochs):
    if hasattr(net, 'update'):
        # print(f'update at epoch: {epoch}')
        net.update(epoch=epoch, trn_dataset=trn_dataset)
    net.train()
       
    if args.use_swa and epoch == args.swa_start:
        net.swa_init()
        
    cum_loss = 0; ctr = 0
    for optim in optims.values(): optim.zero_grad()
    t = tqdm(trn_loader, desc='Epoch: 0, Loss: 0.0', leave=True, disable=not accelerator.is_main_process)
    for i, b in enumerate(t):
        loss = criterion(net, b)
        scaler.scale(loss.float()).backward() if args.use_grad_scaler else accelerator.backward(loss)
        
        if args.use_swa and i % args.swa_step == 0:
            net.swa_step()
            
        for k in optims.keys():
            if i%optim_wrap[k]["accum_steps"] == 0:
                scaler.step(optims[k]) if args.use_grad_scaler else optims[k].step();
                if args.use_grad_scaler: scaler.update()
                schedulers[k].step(); 
                optims[k].zero_grad()

        cum_loss += loss.item()
        ctr += 1
        t.set_description('Epoch: %d/%d, Loss: %.4E'%(epoch, args.num_epochs, (cum_loss/ctr)), refresh=True)
        
    accelerator.print(*[f'{k} lr: {"%.2E"%optims[k].param_groups[0]["lr"]}' for k in optims.keys()])
    if accelerator.is_main_process:
        wandb.log({'epoch': epoch, 'loss': cum_loss/ctr}, step=epoch)
        accelerator.print(f'mean loss after epoch {epoch}/{args.num_epochs}: {"%.4E"%(cum_loss/ctr)}', flush=True)
        if epoch%args.eval_interval == 0 or epoch == (args.num_epochs-1):
            if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                score_mat = net.module.predict(tst_loader)
            else:
                score_mat = net.predict(tst_loader)
            if tst_filter_mat is not None: _filter(score_mat, tst_filter_mat, copy=False)
            metrics = XCMetrics(score_mat, tst_X_Y, inv_prop, method=f'Ep {epoch}', disp=True)
            metrics['loss'] = ["%.4E"%(cum_loss/ctr)]
            metrics.to_csv(open(f'{OUT_DIR}/metrics.tsv', 'a+'), sep='\t', header=(epoch==0))
            
            if metrics.loc[f'Ep {epoch}']['nDCG@5'] > best_ndcg:
                best_ndcg = metrics.loc[f'Ep {epoch}']['nDCG@5']
                print(_c(f'Found new best model with nDCG@5: {"%.2f"%best_ndcg}\n', attr='blue'))
                
                if args.save:
                    sp.save_npz(f'{OUT_DIR}/score_mat.npz', score_mat)
                    net.save_model(f'{OUT_DIR}/model.pt')
               
            wandb.log({k: metrics.loc[f'Ep {epoch}'][k] for k in ['P@1', 'P@5', 'PSP@1', 'PSP@5', 'nDCG@5', 'R@100']}, step=epoch)
            wandb.log({'metrics': metrics}, step=epoch)
        sys.stdout.flush()
    accelerator.wait_for_everyone()

if accelerator.is_main_process:
    # print(net.load_state_dict(torch.load(f'{OUT_DIR}/model.pt', map_location='cpu'), strict=False))
    # net.to(args.device)
    if hasattr(net, 'update'): net.update()
    
    if args.save_embs:
        PLOG('Saving trained embeddings')
        trn_embs = net.get_embs(trn_dataset)
        tst_embs = net.get_embs(tst_dataset)
        np.save(f'{args.OUT_DIR}/trn_embs.npy', trn_embs)
        np.save(f'{args.OUT_DIR}/tst_embs.npy', tst_embs)
        
    if args.ranker:
        PLOG('Sparse ranker')
        net.encoder = net.encoder_bottleneck = None
        trn_embs = np.load(f'{args.OUT_DIR}/trn_embs.npy')
        tst_embs = np.load(f'{args.OUT_DIR}/tst_embs.npy')
        
        trn_dataset = FixedDataset(trn_embs, trn_X_Y)
        tst_dataset = FixedDataset(tst_embs, tst_X_Y)
        trn_embs_loader = torch.utils.data.DataLoader(
            trn_dataset,
            batch_size=args.bsz*4,
            num_workers=4,
            collate_fn=XCCollator(args.numy, trn_dataset),
            shuffle=False,
            pin_memory=True)
        tst_embs_loader = torch.utils.data.DataLoader(
            tst_dataset,
            batch_size=args.bsz*4,
            num_workers=4,
            collate_fn=XCCollator(args.numy, tst_dataset),
            shuffle=False,
            pin_memory=True)
        
        print(_c('Computing top K predictions for ranker...', attr='blue'))
        tst_score_mat = _filter(net.predict(tst_embs_loader, K=args.ranker_topk), tst_filter_mat, copy=False)
        trn_score_mat = net.predict(trn_embs_loader, K=args.ranker_topk)
        XCMetrics(tst_score_mat, tst_X_Y, inv_prop)
        sp.save_npz(f'{args.OUT_DIR}/tst_score_mat.npz', tst_score_mat)
        
        XCMetrics(trn_score_mat, trn_X_Y, inv_prop)
        sp.save_npz(f'{args.OUT_DIR}/trn_score_mat.npz', trn_score_mat)
        
        print(_c('Computing embeddings for ranker...', attr='blue'))
        trn_X_Xf = normalize(sp.load_npz(f'{args.DATA_DIR}/X.trn.npz'), copy=False)
        tst_X_Xf = normalize(sp.load_npz(f'{args.DATA_DIR}/X.tst.npz'), copy=False)
        trn_ranker_embs = sp.hstack((normalize(trn_embs, copy=False), trn_X_Xf)).tocsr().astype(np.float32)
        tst_ranker_embs = sp.hstack((normalize(tst_embs, copy=False), tst_X_Xf)).tocsr().astype(np.float32)
        
        print(_c('Training Ranker...', attr='blue'))
        prob = MLProblem(trn_ranker_embs, trn_X_Y, C=sp.identity(trn_X_Y.shape[1]).tocsr(), M=trn_score_mat, R=None)
        mlm = MLModel.train(prob)
        mlm.pred_params.post_processor = 'l3-hinge'
        tst_shortlist = tst_score_mat.copy(); tst_shortlist.data[:] = 1
        print(_c('Predicting ranker...', attr='blue'))
        tst_ranker_score_mat = mlm.predict(tst_ranker_embs, csr_codes=tst_shortlist.astype(np.float32), only_topk=args.ranker_topk)
        XCMetrics(tst_ranker_score_mat, tst_X_Y, inv_prop)
        sp.save_npz(f'{args.OUT_DIR}/tst_ranker_score_mat.npz', tst_ranker_score_mat)
        
        tst_score_mat.data **= args.ranker_alpha
        tst_ranker_score_mat.data **= (1-args.ranker_alpha)
        combined_score_mat = tst_score_mat.multiply(tst_ranker_score_mat)
        XCMetrics(combined_score_mat, tst_X_Y, inv_prop)
        sp.save_npz(f'{args.OUT_DIR}/tst_combined_score_mat.npz', combined_score_mat)
        print(_c('Done.', attr='blue'))
        
    if args.score_calibrate:
        PLOG('Calibrating ELIAS and sparse ranker scores')
        tst_combined_score_mat = sp.load_npz(f'{args.OUT_DIR}/tst_combined_score_mat.npz')
        val_combined_score_mat = sp.load_npz(f'{args.OUT_DIR}/val_combined_score_mat.npz')
        from sklearn import tree
        def get_tree_fts(smats, tmat = None, clf = None, mode='test'):
            temp = smats[0].tocoo()
            scores = []
            for smat in smats:
                scores.append(np.array(smat[temp.row, temp.col]).reshape(-1, 1))
            scores.append(inv_prop[temp.col].reshape(-1, 1))
            scores = np.hstack(scores)

            if mode == 'train':
                targets = np.array(tmat[temp.row, temp.col]).ravel()
                return scores, targets
            else:
                res = smats[0].copy()
                res[temp.row, temp.col] = clf.predict_proba(scores)[:, 1]
                return res.tocsr()
        scores, targets = get_tree_fts([val_combined_score_mat], tmat=val_X_Y, mode='train')
        clf = tree.DecisionTreeClassifier(max_depth=6)
        clf = clf.fit(scores, targets)
        final_score_mat = get_tree_fts([tst_combined_score_mat], clf=clf, mode='test')*4 + tst_combined_score_mat
        XCMetrics(final_score_mat, tst_X_Y, inv_prop)
        sp.save_npz(f'{args.OUT_DIR}/final_score_mat.npz', final_score_mat)
        
    if args.gen_A_init:
        PLOG('Generating sparse approximate for A')
        net.eval(); net.swa_swap_params()
        numx = len(trn_dataset)
        if accelerator.is_main_process:
            M_indices = torch.zeros((numx, args.beam), dtype=torch.long)
            M_values = torch.zeros((numx, args.beam), dtype=torch.float)

        with torch.no_grad():
            for b in tqdm(trn_loader):
                b = net.ToD(b)
                b_ids = b['ids']
                embs = net.encode(b).detach()
                res = torch.mm(embs, net.module.w1.weight.T) + net.module.w1.bias.reshape(1, -1)
                res = torch.clamp(net.module.alpha*F.softmax(res, dim=1), min=0, max=1)
                topb_clusters = res.topk(args.beam, dim=1)
                
                all_topb_clusters = accelerator.gather(topb_clusters)
                all_b_ids = accelerator.gather(b_ids)
                if accelerator.is_main_process:
                    M_indices[all_b_ids] = all_topb_clusters.indices.cpu()
                    M_values[all_b_ids] = all_topb_clusters.values.cpu()
                    
                del b, res, topb_clusters, all_topb_clusters, all_b_ids
        net.swa_swap_params()
        
        if accelerator.is_main_process:
            trn_Y_X = trn_X_Y.T.tocsr()
            K = M_indices.shape[1]
            data_bin = np.ones(numx * K) 
            data_val = M_values.reshape(-1).cpu().numpy()
            inds = M_indices.reshape(-1).cpu().numpy()
            indptr = np.arange(0, numx*K + 1, K)
            shape = (numx, net.module.w1.weight.shape[0])

            M_bin = sp.csr_matrix((data_bin, inds, indptr), shape)
            M_val = sp.csr_matrix((data_val, inds, indptr), shape)
            C_bin = trn_Y_X.dot(M_bin)
            C_val = trn_Y_X.dot(M_val)

            CT_bin = C_bin.T.tocsr()
            CT_val = C_val.T.tocsr()

            CT_bin_topk = retain_topk(CT_bin, k=args.kappa)
            CT_val_topk = retain_topk(CT_val, k=args.kappa)
            A_init = CT_bin_topk + CT_val_topk
            A_init.sum_duplicates()
            sp.save_npz(f'{args.OUT_DIR}/A_init.npz', A_init)