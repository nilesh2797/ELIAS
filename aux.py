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
# experiment definers
parser.add_argument('--project', default='ELIAS')
parser.add_argument('--expname', default='')
parser.add_argument('--dataset', required=True)
parser.add_argument('--maxlen', type=int, default=32, help='max seq length for transformer')
parser.add_argument('--net', default='elias')
parser.add_argument('--stage', type=int, default=1)
parser.add_argument('--loss', default='joint')
parser.add_argument('--loss-with-logits', action='store_true', default=False)
parser.add_argument('--joint-loss-gamma', type=float, default=0.05)
parser.add_argument('--tf', default='bert-base-uncased')
parser.add_argument('--tf-pooler', default='cls')

# network hyperparameters
parser.add_argument('--C', type=int, default=0)
parser.add_argument('--max-leaf', type=int, default=100)
parser.add_argument('--beam', type=int, default=20)
parser.add_argument('--alpha', type=int, default=10)
parser.add_argument('--beta-gain', type=float, default=1.5)
parser.add_argument('--K', type=int, default=2000)
parser.add_argument('--kappa', type=int, default=1000)

# training hyperparameters
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

# misc. hyperparameters
parser.add_argument('--embs-dir', default='')
parser.add_argument('--save', action='store_true', help='save model and score mat', default=False)
parser.add_argument('--online-tok', action='store_true', help='use online tokenized dataset', default=False)
parser.add_argument('--resume-path', help='continue training from previous checkpoint', default='')
parser.add_argument('--cmat-file', help='saved cluster matrix npz file', default='')
parser.add_argument('--cmat-seed', help='initial cluster matrix seed', default=-1, type=int)
parser.add_argument('--A-init-file', help='saved candidates npz file', default='')
parser.add_argument('--save-embs', action='store_true', default=False)
parser.add_argument('--ranker', action='store_true', default=False)
parser.add_argument('--ranker-alpha', type=float, default=0.5)
parser.add_argument('--ranker-topk', type=int, default=100)
parser.add_argument('--score-calibrate', action='store_true', default=False)

# script options
parser.add_argument('--eval', action='store_true', help='evaluate model', default=False)
parser.add_argument('--eval-topk', type=int, default=100)
parser.add_argument('--gen-cmat', action='store_true', help='generate label partition cluster matrix', default=False)
parser.add_argument('--no-emb-mu', dest='emb_mu', action='store_false', help='dont using dense embeddings for label centroids used in clustering')
parser.add_argument('--no-bow-mu', dest='bow_mu', action='store_false', help='dont using bow features for label centroids used in clustering')
parser.add_argument('--gen-A', action='store_true', help='generate an initialization for A with final model', default=False)

parser.set_defaults(amp_encode=True)
parser.set_defaults(use_swa=False)
parser.set_defaults(cmat_update=True)
parser.set_defaults(emb_mu=True)
parser.set_defaults(bow_mu=True)

args = parser.parse_args()
if args.expname != '':
    if os.path.exists(f'Results/{args.project}/{args.dataset}/{args.expname}/args.json'):
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
        trn_dataset = PreTokBertDataset(f'{DATA_DIR}/{args.token_type}-{args.maxlen}', trn_X_Y, args.maxlen, doc_type='trn')
        tst_dataset = PreTokBertDataset(f'{DATA_DIR}/{args.token_type}-{args.maxlen}', tst_X_Y, args.maxlen, doc_type='tst')
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

accelerator.wait_for_everyone()
PLOG('Initializing Net')
net = NETS[args.net](args)

args.resume_path = f'{OUT_DIR}/model.pt'

if os.path.exists(args.resume_path):
    accelerator.print(f'Loading net state dict from: {args.resume_path}')
    loaded_state = torch.load(args.resume_path, map_location='cpu')
    accelerator.print(net.load_state_dict(loaded_state, strict=True))
else:
    assert not (args.eval or args.gen_A)

net, trn_loader, tst_loader = accelerator.prepare(net, trn_loader, tst_loader)
net_module = net
if isinstance(net, torch.nn.parallel.DistributedDataParallel):
    net.ToD = net.module.ToD
    net.encode = net.module.encode
    net.swa_swap_params = net.module.swa_swap_params
    net_module = net.module
    
net.eval(); net.swa_swap_params()
    
if args.eval:
    K = args.eval_topk
    tst_numx = tst_X_Y.shape[0]
    if accelerator.is_main_process:
        data = np.zeros((tst_numx, K))
        inds = np.zeros((tst_numx, K), dtype=np.int32)
        indptr = np.arange(0, tst_numx*K+1, K)

    with torch.no_grad():
        for b in tqdm(tst_loader, leave=True, desc='Predicting', disable=not accelerator.is_main_process):
            b = net.ToD(b)
            b_ids = b['ids']
            out = net(b)
            if isinstance(out, torch.Tensor): # BxL shaped out
                top_data, top_inds = torch.topk(out, K)
            elif isinstance(out, tuple) and len(out) == 2: # (logits, indices) shaped out 
                top_data, temp_inds = torch.topk(out[0], K)
                top_inds = torch.gather(out[1], 1, temp_inds)
                del temp_inds
            else:
                print(f'Got unsupported type output: {type(out)}: {out}')
                break
                
            all_top_data = accelerator.gather(top_data)
            all_top_inds = accelerator.gather(top_inds)
            all_b_ids = accelerator.gather(b_ids)
            if accelerator.is_main_process:
                data[all_b_ids.cpu().numpy()] = all_top_data.detach().cpu().numpy()
                inds[all_b_ids.cpu().numpy()] = all_top_inds.detach().cpu().numpy()
            del top_data, top_inds, b, out, all_top_data, all_top_inds, all_b_ids

    if accelerator.is_main_process:
        score_mat = sp.csr_matrix((data.ravel(), inds.ravel(), indptr), tst_X_Y.shape)

        # remove padding if any
        if any(score_mat.indices == tst_X_Y.shape[1]):
            score_mat.data[score_mat.indices == tst_X_Y.shape[1]] = 0
            score_mat.eliminate_zeros()
            
        if tst_filter_mat is not None: _filter(score_mat, tst_filter_mat, copy=False)
        metrics = XCMetrics(score_mat, tst_X_Y, inv_prop, method=f'Ep -1', disp=True)
        metrics.to_csv(open(f'{OUT_DIR}/tst_metrics.tsv', 'a+'), sep='\t', header=True)
        sp.save_npz(f'{OUT_DIR}/score_mat.npz', score_mat)
        
if args.gen_cmat:
    assert (args.emb_mu or args.bow_mu)
    
    if args.emb_mu:
        if accelerator.is_main_process:
            emb_mu = np.zeros((args.numy, args.embs_dim))
            num_seen_ids = 0; numx = len(trn_dataset)

        with torch.no_grad():
            
            for b in tqdm(trn_loader, disable=not accelerator.is_main_process, desc='Computing embs mu'):
                b = net.ToD(b)
                all_b_ids = accelerator.gather(b['ids']).detach().cpu().numpy()
                all_embs = accelerator.gather(net.encode(b).contiguous()).detach().cpu().numpy()
               
                if accelerator.is_main_process:
                    bsz = all_b_ids.shape[0]
                    num_seen_ids += bsz
                    # doing this because ddp forcefully makes the last batch of equal batch size
                    if num_seen_ids > numx:
                        num_rel_ids = numx - num_seen_ids + bsz
                        all_b_ids = all_b_ids[:num_rel_ids]
                        all_embs = all_embs[:num_rel_ids]
                        
                    batch_X_Y = trn_X_Y[all_b_ids]
                    all_batch_Y = np.unique(batch_X_Y.indices)
                    batch_X_Y = batch_X_Y[:, all_batch_Y]
                    batch_Y_mu = batch_X_Y.T.tocsr().dot(normalize(all_embs, copy=False))
                    emb_mu[all_batch_Y] += batch_Y_mu
    
    if accelerator.is_main_process:
        mu_list = []
        if args.bow_mu:
            bow_fts = normalize(sp.load_npz(f'{DATA_DIR}/X.trn.npz'), copy=False)
            mu_list.append(trn_X_Y.T.tocsr().dot(bow_fts))
            del bow_fts
        if args.emb_mu:
            mu_list.append(emb_mu)
           
        mu = normalize(sp.hstack(mu_list), copy=False).astype(np.float32)
        cmat = Indexer.gen(mu, 
                           indexer_type="hierarchicalkmeans", 
                           nr_splits=args.C, 
                           max_leaf_size=args.max_leaf, 
                           seed=args.cmat_seed).chain[-1].T
        sp.save_npz(f'{args.OUT_DIR}/cmat.npz', cmat)
    

if args.gen_A:
    PLOG('Generating sparse approximate for A')
    numx = len(trn_dataset)
    if accelerator.is_main_process:
        M_indices = torch.zeros((numx, args.beam), dtype=torch.long)
        M_values = torch.zeros((numx, args.beam), dtype=torch.float)

    with torch.no_grad():
        for b in tqdm(trn_loader, disable=not accelerator.is_main_process, desc='Gen Matching Matrix'):
            b = net.ToD(b)
            b_ids = b['ids']
            embs = net.encode(b).detach()
            res = torch.mm(embs, net_module.w1.weight.T) + net_module.w1.bias.reshape(1, -1)
            res = torch.clamp(net_module.alpha*F.softmax(res, dim=1), min=0, max=1)
            topb_clusters = res.topk(args.beam, dim=1)

            all_topb_clusters = accelerator.gather(topb_clusters)
            all_b_ids = accelerator.gather(b_ids)
            if accelerator.is_main_process:
                M_indices[all_b_ids] = all_topb_clusters.indices.cpu()
                M_values[all_b_ids] = all_topb_clusters.values.cpu()

            del b, res, topb_clusters, all_topb_clusters, all_b_ids

    if accelerator.is_main_process:
        trn_Y_X = trn_X_Y.T.tocsr()
        K = M_indices.shape[1]
        data_bin = np.ones(numx * K) 
        data_val = M_values.reshape(-1).cpu().numpy()
        inds = M_indices.reshape(-1).cpu().numpy()
        indptr = np.arange(0, numx*K + 1, K)
        shape = (numx, net_module.w1.weight.shape[0])

        M_bin = sp.csr_matrix((data_bin, inds, indptr), shape)
        M_val = sp.csr_matrix((data_val, inds, indptr), shape)
        C_bin = trn_Y_X.dot(M_bin)
        C_val = trn_Y_X.dot(M_val)

        CT_bin = C_bin.T.tocsr()
        CT_val = C_val.T.tocsr()

        CT_bin_topk = retain_topk(CT_bin, k=args.kappa)
        CT_val_topk = retain_topk(CT_val, k=args.kappa)
#         A_init = CT_bin_topk + CT_val_topk
        A_init = CT_bin_topk
        A_init.sum_duplicates()
        sp.save_npz(f'{args.OUT_DIR}/A_init.npz', A_init)
        
net.swa_swap_params()