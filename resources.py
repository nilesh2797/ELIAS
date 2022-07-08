#!/usr/bin/env python
# coding: utf-8

import os
import sys
import numpy as np
from operator import itemgetter
import operator
import scipy.sparse as sp
import random as rnd
import seaborn as sns
from numpy import genfromtxt
import pandas as pd
import matplotlib.pyplot as plt
import struct
from sklearn.svm import LinearSVC
import heapq
from sklearn.preprocessing import normalize
from multiprocessing import Pool
from scipy import sparse
from tqdm import tqdm
import torch
import nmslib
import time
from scipy.sparse import csr_matrix
np.random.seed(22)

import xclib.evaluation.xc_metrics as xc_metrics
import xclib.data.data_utils as data_utils

import xclib
from tabulate import tabulate
plt.style.use('dark_background')
from io import StringIO

if '__sysstdout__' not in locals():
    __sysstdout__ = sys.stdout

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

def get_text(x, text, X_Xf, sep=' ', K=-1, attr='bold underline'):
    if K == -1: K = X_Xf[x].nnz
    sorted_inds = X_Xf[x].indices[np.argsort(-X_Xf[x].data)][:K]
    return '%d : \n'%x + sep.join(['%s(%.2f, %d)'%(_c(text[i], attr=attr), X_Xf[x, i], i) for i in sorted_inds])

class Visualize:
    mats = {};
    colors = ['green', 'yellow', 'purple', 'blue', 'red']
    def __init__(self, mats, row_text = None, col_text = None):
        # first mat is base mat
        self.base_mat = list(mats.values())[0]
        self.mats = {k : {'mat' : v, 'color' : self.colors[i], 'base' : i==0} for i, (k, v) in enumerate(mats.items()) }
        self.row_text = row_text
        self.col_text = col_text
        
    def get_row_text(self, x):
        return self.row_text[x]
    
    def get_col_text(self, x):
        return self.col_text[x]
    
    def getX(self, x, K=10):
        print('Raw text \t: %d : %s'%(x, _c(self.get_row_text(x), attr='bold underline')))
        print('Feature text \t: %s'%get_text(x, Xf, sp_tst_X_Xf))
        for name, obj in self.mats.items():
            print(_c('\n' + name + ' : ', attr='bold %s'%(obj['color'])))
            sorted_inds = obj['mat'][x].indices[np.argsort(-obj['mat'][x].data)[:K]]
            for i, ind in enumerate(sorted_inds):
                attr = 'ENDC'
                if self.base_mat[x, ind] and not obj['base']: attr = 'reverse'
                print(_c('%d : %s[%d] : %.4f'%(i+1, self.get_col_text(ind), ind, obj['mat'][x, ind]), attr=attr))

class CaptureIO(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.append(''.join(self._stringio.getvalue()))
        del self._stringio    # free up some memory
        sys.stdout = self._stdout
        
def dump_output_log(filename = None, clear=False, overwrite=False):
    if filename is None: filename = '%s/visual_analysis.log'%results_dir
    global output
    mode= 'a+'
    if overwrite: mode = 'w'
    
    with open(filename, mode) as f:
        f.write(';'.join([*output, '']))
        
    if clear:
        with CaptureIO() as output: print('init')
            
def load_output_log(filename = None):
    if filename is None: filename = '%s/visual_analysis.log'%results_dir    
    log = None
    with open(filename, 'r') as f:
        log = f.read().split(';')
    return log

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
        
def recall(spmat, X_Y = None, K = [1, 10, 50, 100]):
    if X_Y is None : X_Y = tst_X_Y.copy()
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
            
def printacc(score_mat, X_Y = None, K = 5, disp = True, inv_prop_ = -1):
    if X_Y is None: X_Y = tst_X_Y
    if inv_prop_ is -1 : inv_prop_ = inv_prop
        
    acc = xc_metrics.Metrics(X_Y.tocsr().astype(np.bool), inv_prop_)
    metrics = np.array(acc.eval(score_mat, K))*100
    df = pd.DataFrame(metrics)
    
    if inv_prop_ is None : df.index = ['P', 'nDCG']
    else : df.index = ['P', 'nDCG', 'PSP', 'PSnDCG']
        
    df.columns = [i+1 for i in range(K)]
    if disp: print(df.round(2))
    return df

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

def XCMetrics(score_mat, X_Y, inv_prop, disp = True, fname = None, method = 'Method'): 
    X_Y = X_Y.tocsr().astype(np.bool_)
    acc = xc_metrics.Metrics(X_Y, inv_prop)
    xc_eval_metrics = np.array(acc.eval(score_mat, 5))*100
    xc_eval_metrics = pd.DataFrame(xc_eval_metrics)
    
    if inv_prop is None : xc_eval_metrics.index = ['P', 'nDCG']
    else : xc_eval_metrics.index = ['P', 'nDCG', 'PSP', 'PSnDCG']    
    xc_eval_metrics.columns = [(i+1) for i in range(5)]
    
    rank_mat = sp_rank(score_mat)
    intrsxn_mat = rank_mat.multiply(X_Y)
    ret_eval_metrics = pd.DataFrame({'R': Recall(intrsxn_mat, X_Y, K=[10, 20, 100]), 'MRR': MRR(intrsxn_mat, X_Y, K=[10])}).T
    ret_eval_metrics = ret_eval_metrics.reindex(sorted(ret_eval_metrics.columns), axis=1)
        
    df1 = xc_eval_metrics[[1,3,5]].iloc[[0,1,2]].round(2).stack().to_frame().transpose()
    df2 = ret_eval_metrics.iloc[[0,1]].round(2).stack().to_frame().transpose()

    df = pd.concat([df1, df2], axis=1)
    df.columns = [f'{col[0]}@{col[1]}' for col in df.columns.values]
    df.index = [method]

    if disp:
        disp_df = df[['P@1', 'P@3', 'P@5', 'nDCG@1', 'nDCG@3', 'nDCG@5', 'PSP@1', 'PSP@3', 'PSP@5', 'R@10', 'R@20', 'R@100']].round(2)
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

def vis_point(x, spmat, X, Y, nnz, true_mat=None, sep='', K=-1, expand=False, trnx_nnz=None):
    if K == -1: K = spmat[x].nnz
    if true_mat is None: true_mat = tst_X_Y
        
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

def decileWisePVolume(score_mats, tst_X_Y, K=5, mask=None):
    if mask is None : mask = decile_mask
    plt.xticks(range(10))
    plt.title('decile contribution to P@%d'%(K))
        
    alldata = []
    for score_mat, name in score_mats:
        temp_score_mat = score_mat.copy()
        temp_score_mat = xclib.utils.sparse.retain_topk(temp_score_mat, k=K)
        intrsxn_score_mat = temp_score_mat.multiply(tst_X_Y)
        
        intrsxn_data = [mask[i, intrsxn_score_mat.indices].sum()*100 / (intrsxn_score_mat.shape[0]*K) for i in range(10)]
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

def decileWiseVolume(score_mats, tst_X_Y, K=5, mask=None):
    if mask is None : mask = decile_mask
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