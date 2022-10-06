#!/usr/bin/env python
# coding: utf-8

import os, sys, yaml, argparse, re
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import csr_matrix

import xclib
import xclib.evaluation.xc_metrics as xc_metrics

np.random.seed(22)

if '__sysstdout__' not in locals():
    __sysstdout__ = sys.stdout

def load_yaml(fname):
    yaml_dict = yaml.safe_load(open(fname))
    yaml_dict_list = []
    if '__dependency__' in yaml_dict:
        yaml_dict['__dependency__'] = ', '.join([dep_fname if os.path.isabs(dep_fname) else f'{os.getcwd()}/{dep_fname.strip()}' for dep_fname in yaml_dict['__dependency__'].split(',')])
        for dep_fname in yaml_dict['__dependency__'].split(','):
            yaml_dict_list = yaml_dict_list + load_yaml(dep_fname.strip())
    yaml_dict_list.append(yaml_dict)
    return yaml_dict_list

def load_config_and_runtime_args(argv, **extra_args):
    try: config_sep_index = [x.startswith('-') for x in argv[1:]].index(True) 
    except: config_sep_index = len(argv[1:])
    config_args = argv[1:1+config_sep_index]
    runtime_args = argv[1+config_sep_index:]

    parser = argparse.ArgumentParser()
    yaml_dict_lol = [load_yaml(fname) for fname in config_args]
    yaml_dict_list = [yaml_dict for yaml_dict_list in yaml_dict_lol for yaml_dict in yaml_dict_list]
    yaml_dict_list.append(extra_args)
    config = {k: v for d in yaml_dict_list for k, v in d.items()}
    config = pd.json_normalize(config, sep='_').to_dict(orient="records")[0]
    for k, v in config.items():
        parser.add_argument(f'--{k}', default=v, type=str_to_bool if isinstance(v, bool) else type(v))
    args = parser.parse_args(runtime_args)
    args.__dict__ = {k: re.sub(r'\[(\w+)\]', lambda x: args.__dict__[x.group(0)[1:-1]], v) if isinstance(v, str) else v for k, v in args.__dict__.items()}
    return args

def get_inv_prop(X_Y, dataset_name):
    if "amazon" in dataset_name.lower(): A = 0.6; B = 2.6
    elif "wiki" in dataset_name.lower() and "wikiseealso" not in dataset_name.lower(): A = 0.5; B = 0.4
    else : A = 0.55; B = 1.5
    return xc_metrics.compute_inv_propesity(X_Y, A, B)

def load_filter_mat(fname, shape):
    filter_mat = None
    if os.path.exists(fname):
        temp = np.fromfile(fname, sep=' ').astype(int)
        temp = temp.reshape(-1, 2).T
        filter_mat = sp.coo_matrix((np.ones(temp.shape[1]), (temp[0], temp[1])), shape).tocsr()
    return filter_mat

def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def dump_diff_config(config_fname, config_dict):
    if os.path.exists(config_fname):
        with open(config_fname, 'a+') as f:
            print('# New experiment', file=f)
            prev_config = yaml.safe_load(open(config_fname))
            diff_config = dict(set(config_dict.items()) - set(prev_config.items()))
            print('', file=f)
            if len(diff_config) > 0:
                yaml.safe_dump(diff_config, f)
                return diff_config
    else:
        yaml.safe_dump(config_dict, open(config_fname, 'w'))

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def get_text(x, text, X_Xf, sep=' ', K=-1, attr='bold underline'):
    if K == -1: K = X_Xf[x].nnz
    sorted_inds = X_Xf[x].indices[np.argsort(-X_Xf[x].data)][:K]
    return '%d : \n'%x + sep.join(['%s(%.2f, %d)'%(_c(text[i], attr=attr), X_Xf[x, i], i) for i in sorted_inds])

import decimal
def myprint(*args, sep = ' ', end = '\n'):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args), sep = sep, end = end)

def drange(x, y, jump):
    x = decimal.Decimal(x)
    y = decimal.Decimal(y)
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)
        
def recall(spmat, X_Y, K = [1, 10, 50, 100]):
    X_Y.data[:] = 1
    ans = {}
    rank_mat = xclib.utils.sparse.rank(spmat)
    
    for k in K:
        temp = rank_mat.copy()
        temp.data[temp.data > k] = 0.0
        temp.eliminate_zeros()
        temp.data[:] = 1.0
        intrsxn = temp.multiply(X_Y)

        num = np.array(intrsxn.sum(axis=1)).ravel()
        den = np.maximum(np.array(X_Y.sum(axis=1)).ravel(), 1)

        recall = (num/den).mean()
        ans['R@%d'%k] = recall*100
    
    return ans

def _filter(score_mat, filter_mat, copy=True):
    if filter_mat is None:
        return score_mat
    if copy:
        score_mat = score_mat.copy()
    
    temp = filter_mat.tocoo()
    score_mat[temp.row, temp.col] = 0
    del temp
    score_mat = score_mat.tocsr()
    score_mat.eliminate_zeros()
    return score_mat

def get_sorted_spmat(spmat):
    coo = spmat.tocoo()
    temp = np.array([coo.col, -coo.data, coo.row])
    temp = temp[:, np.lexsort(temp)]
    del coo

    inds, cnts = np.unique(temp[2].astype(np.int32), return_counts=True)
    indptr = np.zeros_like(spmat.indptr)
    indptr[inds+1] = cnts
    indptr = np.cumsum(indptr)

    new_spmat = csr_matrix((-temp[1], temp[0].astype(np.int32), indptr), (spmat.shape))
    del inds, cnts, indptr, temp
    return new_spmat
                
def write_sparse_mat(X, filename, header=True):
    if not isinstance(X, csr_matrix):
        X = X.tocsr()
    X.sort_indices()
    with open(filename, 'w') as f:
        if header:
            print("%d %d" % (X.shape[0], X.shape[1]), file=f)
        for y in X:
            idx = y.__dict__['indices']
            val = y.__dict__['data']
            sentence = ' '.join(['%d:%.5f'%(x, v)
                                 for x, v in zip(idx, val)])
            print(sentence, file=f)
            
def read_sparse_mat(filename, use_xclib=True):
    if use_xclib:
        return xclib.data.data_utils.read_sparse_file(filename)
    else:
        with open(filename) as f:
            nr, nc = map(int, f.readline().split(' '))
            data = []; indices = []; indptr = [0]
            for line in tqdm(f):
                if len(line) > 1:
                    row = [x.split(':') for x in line.split()]
                    tempindices, tempdata = list(zip(*row))
                    indices.extend(list(map(int, tempindices)))
                    data.extend(list(map(float, tempdata)))
                    indptr.append(indptr[-1]+len(tempdata))
                else:
                    indptr.append(indptr[-1])
            score_mat = csr_matrix((data, indices, indptr), (nr, nc))
            del data, indices, indptr
            return score_mat

from xclib.utils.sparse import rank as sp_rank

def _topk(rank_mat, K, inplace=False):
    topk_mat = rank_mat if inplace else rank_mat.copy()
    topk_mat.data[topk_mat.data > K] = 0
    topk_mat.eliminate_zeros()
    return topk_mat

def Recall(rank_intrsxn_mat, true_mat, K=[1,3,5,10,20,50,100]):
    K = sorted(K, reverse=True)
    topk_intrsxn_mat = rank_intrsxn_mat.copy()
    res = {}
    for k in K:
        topk_intrsxn_mat = _topk(topk_intrsxn_mat, k, inplace=True)
        res[k] = (topk_intrsxn_mat.getnnz(1)/true_mat.getnnz(1)).mean()*100.0
    return res

def MRR(rank_intrsxn_mat, true_mat, K=[1,3,5,10,20,50,100]):
    K = sorted(K, reverse=True)
    topk_intrsxn_mat = _topk(rank_intrsxn_mat, K[0], inplace=True)
    rr_topk_intrsxn_mat = topk_intrsxn_mat.copy()
    rr_topk_intrsxn_mat.data = 1/rr_topk_intrsxn_mat.data
    max_rr = rr_topk_intrsxn_mat.max(axis=1).toarray().ravel()
    res = {}
    for k in K:
        max_rr[max_rr < 1/k] = 0.0
        res[k] = max_rr.mean()*100
    return res

def compute_xmc_metrics(score_mat, X_Y, inv_prop, K=100, disp = True, fname = None, name = 'Method'): 
    X_Y = X_Y.tocsr().astype(np.bool_)
    acc = xc_metrics.Metrics(X_Y, inv_prop)
    xc_eval_metrics = np.array(acc.eval(score_mat, 5))*100
    xc_eval_metrics = pd.DataFrame(xc_eval_metrics)
    
    if inv_prop is None : xc_eval_metrics.index = ['P', 'nDCG']
    else : xc_eval_metrics.index = ['P', 'nDCG', 'PSP', 'PSnDCG']    
    xc_eval_metrics.columns = [(i+1) for i in range(5)]
    
    rank_mat = sp_rank(score_mat)
    intrsxn_mat = rank_mat.multiply(X_Y)
    recallKs = [*[10, 20, 50], *[100*i for i in range(1, 1+(K//100))]]
    ret_eval_metrics = pd.DataFrame({'R': Recall(intrsxn_mat, X_Y, K=recallKs), 'MRR': MRR(intrsxn_mat, X_Y, K=[10])}).T
    ret_eval_metrics = ret_eval_metrics.reindex(sorted(ret_eval_metrics.columns), axis=1)
        
    df1 = xc_eval_metrics[[1,3,5]].iloc[[0,1,2]].round(2).stack().to_frame().transpose()
    df2 = ret_eval_metrics.iloc[[0,1]].round(2).stack().to_frame().transpose()

    df = pd.concat([df1, df2], axis=1)
    df.columns = [f'{col[0]}@{col[1]}' for col in df.columns.values]
    df.index = [name]

    if disp:
        disp_df = df[[*['P@1', 'P@3', 'P@5', 'nDCG@1', 'nDCG@3', 'nDCG@5', 'PSP@1', 'PSP@3', 'PSP@5'], *[x for x in df.columns if x.startswith('R@')]]].round(2)
        print(disp_df.to_csv(sep='\t', index=False))
        print(disp_df.to_csv(sep=' ', index=False))
    if fname is not None:
        if os.path.splitext(fname)[-1] == '.json': df.to_json(fname)
        elif os.path.splitext(fname)[-1] == '.csv': df.to_csv(fname)  
        elif os.path.splitext(fname)[-1] == '.tsv': df.to_csv(fname, sep='\t')  
        else: print(f'ERROR: File extension {os.path.splitext(fname)[-1]} in {fname} not supported')
    return df
    
class bcolors:
    purple = '\033[95m'
    blue = '\033[94m'
    green = '\033[92m'
    warn = '\033[93m' # dark yellow
    fail = '\033[91m' # dark red
    white = '\033[37m'
    yellow = '\033[33m'
    red = '\033[31m'
    
    ENDC = '\033[0m'
    bold = '\033[1m'
    underline = '\033[4m'
    reverse = '\033[7m'
    
    on_grey = '\033[40m'
    on_yellow = '\033[43m'
    on_red = '\033[41m'
    on_blue = '\033[44m'
    on_green = '\033[42m'
    on_magenta = '\033[45m'
    
def _c(*args, attr='bold'):
    string = ''.join([bcolors.__dict__[a] for a in attr.split()])
    string += ' '.join([str(arg) for arg in args])+bcolors.ENDC
    return string

def vis_point(x, spmat, X, Y, nnz, true_mat, sep='', K=-1, expand=False, trnx_nnz=None, trn_Y_X=None, trnX=None):
    if K == -1: K = spmat[x].nnz
        
    sorted_inds = spmat[x].indices[np.argsort(-spmat[x].data)][:K]
    print(f'x[{x}]: {_c(X[x], attr="bold")}\n')
    for i, ind in enumerate(sorted_inds):
        myattr = ""
        if true_mat[x, ind] > 0.1: myattr="yellow"
        print(f'{i+1}) {_c(Y[ind], attr=myattr)} [{ind}] ({"%.4f"%spmat[x, ind]}, {nnz[ind]})')
        if expand:
            for j, trn_ind in enumerate(trn_Y_X[ind].indices[:10]):
                print(f'\t{j+1}) {_c(trnX[trn_ind], attr="green")} [{trn_ind}] ({trnx_nnz[trn_ind]})')
        print(sep)

def get_decile_mask(X_Y):
    nnz = X_Y.getnnz(0)
    sorted_inds = np.argsort(-nnz)
    cumsum_sorted_nnz = nnz[sorted_inds].cumsum()

    deciles = [sorted_inds[np.where((cumsum_sorted_nnz > i*nnz.sum()/10) & (cumsum_sorted_nnz <= (i+1)*nnz.sum()/10))[0]] for i in range(10)]
    decile_mask = np.zeros((10, X_Y.shape[1]), dtype=np.bool)
    for i in range(10):
        decile_mask[i, deciles[i]] = True
    return decile_mask

def decileWisePVolume(score_mats, tst_X_Y, decile_mask, K=5):
    plt.xticks(range(10))
    plt.title('decile contribution to P@%d'%(K))
        
    alldata = []
    for score_mat, name in score_mats:
        temp_score_mat = score_mat.copy()
        temp_score_mat = xclib.utils.sparse.retain_topk(temp_score_mat, k=K)
        intrsxn_score_mat = temp_score_mat.multiply(tst_X_Y)
        
        intrsxn_data = [decile_mask[i, intrsxn_score_mat.indices].sum()*100 / (intrsxn_score_mat.shape[0]*K) for i in range(10)]
        plt.plot(intrsxn_data, label=name)
        plt.legend()
        
        intrsxn_data.append(sum(intrsxn_data))
        alldata.append(intrsxn_data)
        del intrsxn_score_mat, temp_score_mat
        
    df = pd.DataFrame(alldata)
    df.index = [score_mat[1] for score_mat in score_mats]
    df.columns = [*[str(i+1) for i in range(10)], 'P@%d'%K]
    df = df.round(2)
    return df

def decileWiseVolume(score_mats, decile_mask, K=5):
    plt.xticks(range(10))
    plt.title(f'% deciles present in score_mat top {K}')
    
    alldata = []
    for score_mat, name in score_mats:
        temp_score_mat = score_mat.copy()
        temp_score_mat = xclib.utils.sparse.retain_topk(temp_score_mat, k=K)
        data = [decile_mask[i, temp_score_mat.indices].sum()*100 / temp_score_mat.nnz for i in range(10)]
        plt.plot(data, label=name)
        plt.legend()
        alldata.append(data)
        del temp_score_mat
        
    df = pd.DataFrame(alldata)
    df.index = [score_mat[1] for score_mat in score_mats]
    df.columns = [str(i+1) for i in range(10)]
    df = df.round(2)
    return df