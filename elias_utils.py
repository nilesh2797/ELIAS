from accelerate import Accelerator
accelerator = Accelerator()
IS_MAIN_PROC = accelerator.is_main_process

import os, sys, yaml, logging
import logging.config
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from pecos.xmc import MLProblem, MLModel, Indexer

from nets import *
from xclib.utils.sparse import retain_topk
from resources import _c, load_config_and_runtime_args
from datasets import XMCDataManager, XMCEvaluator
from dl_helper import unwrap

import transformers
transformers.set_seed(42)
if torch.__version__ > "1.11":
    torch.backends.cuda.matmul.allow_tf32 = True

# Config and runtime argument parsing
mode = sys.argv[1]
args = load_config_and_runtime_args(sys.argv[1:], no_model=False)
    
args.device = str(accelerator.device)
args.amp = accelerator.state.use_fp16
args.num_gpu = accelerator.state.num_processes
args.DATA_DIR = DATA_DIR = f'Datasets/{args.dataset}'
args.OUT_DIR = OUT_DIR = args.OUT_DIR if hasattr(args, 'OUT_DIR') else f'Results/{args.project}/{args.dataset}/{args.expname}'
args.wandb_id = args.wandb_id if hasattr(args, 'wandb_id') else 'None'
args.resume_path = f'{args.OUT_DIR}/model.pt'
if mode == 'gen_cluster_A' or mode == 'gen_approx_A':
    args.num_val_points = 0
os.makedirs(OUT_DIR, exist_ok=True)

if IS_MAIN_PROC:
    with open('configs/logging.yaml') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = f"{args.OUT_DIR}/{log_config['handlers']['file_handler']['filename']}"
        logging.config.dictConfig(log_config)

logging.info(f'Starting {" ".join(sys.argv)}')
logging.info(f'Experiment name: {_c(args.expname, attr="blue")}, Dataset: {_c(args.dataset, attr="blue")}')
logging.info(f'Wandb ID: {args.wandb_id}')

# Data loading
data_manager = XMCDataManager(args)
trn_loader, val_loader, tst_loader = data_manager.build_data_loaders()
trn_X_Y = trn_loader.dataset.labels.astype(np.float32)

accelerator.wait_for_everyone()
if os.path.exists(args.resume_path) and (not args.no_model):
    net = NETS[args.net](args)
    logging.info(f'Loading net state dict from: {args.resume_path}')
    logging.info(net.load(args.resume_path))
    net, trn_loader, val_loader, tst_loader = accelerator.prepare(net, trn_loader, val_loader, tst_loader)
    net.eval()

if mode == 'gen_cluster_A':
    trn_embs = []
    if os.path.exists(args.resume_path) and (not args.no_model):
        trn_embs.append(normalize(unwrap(net).get_embs(trn_loader, accelerator=accelerator)))
    if IS_MAIN_PROC:
        trn_embs.append(normalize(sp.load_npz(f'{args.DATA_DIR}/X.trn.npz'), copy=False))
        trn_embs = sp.hstack(trn_embs) if len(trn_embs) > 1 else trn_embs[0]
        lbl_centroids = normalize(trn_X_Y.T.tocsr().dot(trn_embs).astype(np.float32), copy=False)

        cmat = Indexer.gen(lbl_centroids, 
                           indexer_type="hierarchicalkmeans", 
                           nr_splits=int(2**np.ceil(np.log2(args.numy/args.max_leaf))), 
                           max_leaf_size=args.max_leaf, 
                           seed=args.cmat_seed).chain[-1].T
        sp.save_npz(f'{args.OUT_DIR}/cmat.npz', cmat)

if mode == 'gen_approx_A':
    if IS_MAIN_PROC:
        M_inds = torch.zeros((len(trn_loader.dataset), args.beam_size), dtype=torch.long)
        M_vals = torch.zeros((len(trn_loader.dataset), args.beam_size), dtype=torch.float)

    with torch.no_grad():
        for b in tqdm(trn_loader, disable=not IS_MAIN_PROC, desc='Gen Matching Matrix'):
            b = unwrap(net).ToD(b)
            embs = unwrap(net).encode(b)
            cluster_scores = torch.clamp(unwrap(net).alpha*F.softmax(unwrap(net).WC(embs), dim=1), min=0, max=1)
            topb_cluster_inds, topb_cluster_vals = unwrap(net).get_cluster_shortlist(cluster_scores, b)

            topb_cluster_inds = accelerator.gather(topb_cluster_inds)
            topb_cluster_vals = accelerator.gather(topb_cluster_vals)
            b['ids'] = accelerator.gather(b['ids'])
            if IS_MAIN_PROC:
                M_inds[b['ids']] = topb_cluster_inds.cpu()
                M_vals[b['ids']] = topb_cluster_vals.cpu()

    if IS_MAIN_PROC:
        trn_Y_X = trn_X_Y.T.tocsr()
        K = M_inds.shape[1]
        data_val = M_vals.reshape(-1).cpu().numpy()
        inds = M_inds.reshape(-1).cpu().numpy()
        indptr = np.arange(0, M_inds.shape[0]*K + 1, K)
        shape = (M_inds.shape[0], unwrap(net).WC.weight.shape[0])

        M_val = sp.csr_matrix((data_val, inds, indptr), shape)
        C_val = trn_Y_X.dot(M_val)
        CT_val = C_val.T.tocsr()

        A_approx = retain_topk(CT_val, k=args.kappa)
        sp.save_npz(f'{args.OUT_DIR}/A_approx.npz', A_approx)

if mode == 'sparse_ranker':
    trn_score_mat = unwrap(net).predict(trn_loader, K=args.eval_topk, accelerator=accelerator)
    val_score_mat = unwrap(net).predict(val_loader, K=args.eval_topk, accelerator=accelerator)
    tst_score_mat = unwrap(net).predict(tst_loader, K=args.eval_topk, accelerator=accelerator)

    trn_embs = unwrap(net).get_embs(trn_loader, accelerator=accelerator)
    val_embs = unwrap(net).get_embs(val_loader, accelerator=accelerator)
    tst_embs = unwrap(net).get_embs(tst_loader, accelerator=accelerator)

    if IS_MAIN_PROC:
        trn_X_Xf, val_X_Xf, tst_X_Xf = data_manager.load_bow_fts(normalize=True)
        trn_ranker_embs = sp.hstack((normalize(trn_embs, copy=False), trn_X_Xf)).tocsr().astype(np.float32)
        val_ranker_embs = sp.hstack((normalize(val_embs, copy=False), val_X_Xf)).tocsr().astype(np.float32)
        tst_ranker_embs = sp.hstack((normalize(tst_embs, copy=False), tst_X_Xf)).tocsr().astype(np.float32)

        trn_labels = trn_loader.dataset.labels.astype(np.float32)
        prob = MLProblem(trn_ranker_embs, trn_labels, C=sp.identity(trn_labels.shape[1]).tocsr(), M=trn_score_mat, R=None)
        mlm = MLModel.train(prob)
        mlm.pred_params.post_processor = 'l3-hinge'
        val_ranker_score_mat = mlm.predict(val_ranker_embs, csr_codes=val_score_mat.astype(np.float32), only_topk=val_score_mat.shape[1])    
        tst_ranker_score_mat = mlm.predict(tst_ranker_embs, csr_codes=tst_score_mat.astype(np.float32), only_topk=tst_score_mat.shape[1])

        if args.ranker_calibrate:
            nnz = trn_labels.getnnz(0)
            from sklearn import tree
            def process_tree_fts(smats, tmat = None, clf = None, mode='test'):
                temp = smats[0].tocoo()
                scores = []
                for smat in smats:
                    scores.append(np.array(smat[temp.row, temp.col]).reshape(-1, 1))
                scores.append(nnz[temp.col].reshape(-1, 1))
                scores = np.hstack(scores)

                if mode == 'train':
                    targets = np.array(tmat[temp.row, temp.col]).ravel()
                    return scores, targets
                else:
                    res = smats[0].copy()
                    res[temp.row, temp.col] = clf.predict_proba(scores)[:, 1]
                    return res.tocsr()
            clf = tree.DecisionTreeClassifier(max_depth=5)
            clf = clf.fit(*process_tree_fts([val_ranker_score_mat, val_score_mat], tmat=val_loader.dataset.labels, mode='train'))
            tst_ranker_score_mat = process_tree_fts([tst_ranker_score_mat, tst_score_mat], clf=clf, mode='test')*0.3 + tst_ranker_score_mat*0.7

        evaluator = XMCEvaluator(args, tst_loader, data_manager, prefix='tst_ranker')
        metrics = evaluator.eval(tst_ranker_score_mat)
        logging.info('\n'+metrics.to_csv(sep='\t', index=False))
        sp.save_npz(f'{args.OUT_DIR}/tst_ranker_score_mat.npz', tst_ranker_score_mat)

accelerator.wait_for_everyone()