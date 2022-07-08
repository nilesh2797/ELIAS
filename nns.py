#!/usr/bin/env python
# coding: utf-8

# ## Exact Search

# ### CPU code
import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import nmslib
import time
from tqdm import tqdm
import xclib

def getnns_cpu(self):
    if self.hp['metric'] != 'ip':
        print('%s not supported.'%self.hp['metric'])
        return 
    
    self.initialize()
    start = time.time()
    for i in tqdm(range(0, self.num_query, self.batch_size)):
        query_slice = self.hp['query'][i:i+self.batch_size]
        prod = np.dot(query_slice, self.w)
        self.indices[i:i+self.batch_size] = np.argsort(prod, axis=1)[:, -self.K:]
        self.data[i:i+self.batch_size] = np.take_along_axis(prod, self.indices[i:i+self.batch_size], axis=1)
    end = time.time()

    print('Total time, time per point : %.2fs, %.4fms/pt'%(end-start, (end-start)*1000/self.num_query))
    return self.getnns(self.hp['data'].shape[0])


# ### GPU code

# In[1]:


def getnns_gpu(self, shard_size=1000000):
    if self.w.shape[0] > shard_size:
        print(f'Doing nns in {(self.w.shape[0]+shard_size-1)//shard_size} shards')
        score_mat = csr_matrix((self.num_query, 0))

        for ctr in tqdm(range(0, self.w.shape[0], shard_size)):
            temp = self.getnns_gpu_shard(range(ctr, min(ctr+shard_size, self.w.shape[0])))
            score_mat = temp if ctr == 0 else xclib.utils.sparse.retain_topk(sp.hstack((score_mat, temp)).tocsr(), k=self.K)
        return score_mat
    else:
        return self.getnns_gpu_shard(range(self.w.shape[0]))

def getnns_gpu_shard(self, shard=None):
    device = self.device
    with torch.no_grad():
        w_gpu = torch.from_numpy(self.w[shard]).float().to(device)

        start = time.time()
        for i in tqdm(range(0, self.num_query, self.batch_size)):
            query_slice_gpu = torch.from_numpy(self.hp['query'][i:i+self.batch_size]).float().to(device)
            
            prod_gpu = None
            if self.hp['metric'] == 'ip':
                prod_gpu = torch.matmul(w_gpu, query_slice_gpu.T).T
            elif self.hp['metric'] == 'euclid':
                prod_gpu = torch.cdist(query_slice_gpu, w_gpu)
                
            batch_data_gpu, batch_indices_gpu = torch.topk(prod_gpu, k=self.K, sorted=True, largest=self.hp['sim'])
            self.data[i:i+self.batch_size], self.indices[i:i+self.batch_size] = batch_data_gpu.cpu().numpy(), batch_indices_gpu.cpu().numpy()
        end = time.time()

        print('Total time, time per point : %.2fs, %.4f ms/pt'%(end-start, (end-start)*1000/self.num_query))
        del w_gpu, query_slice_gpu, prod_gpu, batch_data_gpu, batch_indices_gpu
#         torch.cuda.empty_cache()
        return self.getnns(len(shard))


# In[3]:


def getnns_shorty_gpu(self, shorty):
    if self.hp['metric'] != 'ip':
        print('%s not supported.'%self.hp['metric'])
        return 
    
    self.K = self.hp['K']
    device = self.device
    res = shorty.copy().tocoo()
    
    with torch.no_grad():
        w_gpu = torch.from_numpy(self.hp['data']).float().to(device)
        query_gpu = torch.from_numpy(self.hp['query']).float().to(device)
        rows = res.row
        cols = res.col
        bsz = self.batch_size

        start = time.time()
        for i in tqdm(range(0, res.nnz, bsz)):
            prod_gpu = (query_gpu[rows[i:i+bsz]] * w_gpu[cols[i:i+bsz]]).sum(dim=1)
            res.data[i:i+bsz] = prod_gpu.detach().cpu().numpy()                              
        end = time.time()

        print('Total time, time per point : %.2fs, %.4f ms/pt'%(end-start, (end-start)*1000/self.num_query))
        del w_gpu, query_gpu, prod_gpu
        torch.cuda.empty_cache()
        return res.tocsr()


# ### Wrapper class

# In[6]:


class exact_search:
    hp = {
                'batch_size' : 512,
                'score_mat' : '%s/score_mat_exact.bin'%('.'),
                'data' : None,
                'query' : None,
                'K' : 10,
                'sim' : True,
                'metric' : 'ip',
                'device': 'cuda:0'
              }
    getnns_gpu = getnns_gpu
    getnns_cpu = getnns_cpu
    getnns_gpu_shard = getnns_gpu_shard
    getnns_shorty_gpu = getnns_shorty_gpu
    num_query = None; num_base = None; batch_size = None; w = None; K = None; data = None; indices = None; indptr = None
    device = None
    def __init__(self, hp):
        for k, v in hp.items():
            self.hp[k] = v
        self.initialize()
        
    def initialize(self):
        self.num_query = self.hp['query'].shape[0]
        self.num_base = self.hp['data'].shape[0]
        self.batch_size = self.hp['batch_size']
        self.device = self.hp['device']
        
        if self.hp['metric'] == 'cosine':
            self.hp['query'] = normalize(self.hp['query'], axis=1)
            self.hp['data'] = normalize(self.hp['data'], axis=1)
            self.hp['metric'] = 'ip'
        
        if self.hp['metric'] == 'ip':
            self.w = self.hp['data']
            self.hp['sim'] = True
        elif self.hp['metric'] == 'euclid':
            self.w = self.hp['data']
            self.hp['sim'] = False
            
        self.K = self.hp['K']
        self.data = np.zeros((self.num_query, self.K))
        self.indices = np.zeros((self.num_query, self.K), dtype=int)
        self.indptr = range(0, self.data.shape[0]*self.data.shape[1]+1, self.data.shape[1])
        
    def getnns(self, nc, save = False):
        score_mat = csr_matrix((self.data.ravel(), self.indices.ravel(), self.indptr), (self.num_query, nc))
        if save: 
            sparse.save_npz(self.hp['score_mat'], score_mat)
#         del self.data, self.indptr, self.indices
        return score_mat


# ## HNSW

# ### Training

# In[16]:


def train_hnsw(self):
    start = time.time()
    self.index = nmslib.init(method='hnsw', space=self.hp['metric'])
    self.index.addDataPointBatch(self.hp['data'])
    self.index.createIndex({'M': self.hp['M'], 'indexThreadQty': self.hp['t'], 'efConstruction': self.hp['efC']}, print_progress=True)
    end = time.time()
    
    self.train_time = (end-start)
    print('Training time of ANNS datastructure = %f'%(self.train_time))
    nmslib.saveIndex(self.index, self.hp['model_file'])
    self.model_size = os.path.getsize(self.hp['model_file'])/1e6
    print("Model size : %.2f MBytes"%(self.model_size))


# ### Prediction

# In[17]:


def search_hnsw(self):
    self.index.setQueryTimeParams({'efSearch': self.hp['efS'], 'algoType': 'old'})
    start = time.time()
    nbrs = np.array(self.index.knnQueryBatch(self.hp['query'], k=self.hp['K'], num_threads = self.hp['t']), dtype=object)
    end = time.time()
    
    self.search_time = end-start
    print('Time taken to find approx nearest neighbors = %f'%(self.search_time))
    
    self.data = 1-nbrs[:, 1].ravel()
    self.indptr = range(0, self.data.shape[0]+1, self.hp['K'])
    self.indices = nbrs[:, 0].ravel()
    del nbrs
    return self.getnns()


# ### Wrapper class

# In[77]:


class hnsw_search:
    hp = {
                'metric' : 'cosinesimil', 
                'M' : 50,
                't' : 6,
                'efC' : 100,
                'model_file' : '%s/hnsw.bin'%('.'),
                'data' : None,
                'query' : None,
                'efS' : 100,
                'K' : 10,
                'score_mat' : '%s/score_mat_hnsw.bin'%('.'),
                'name' : ''
              }
    train = train_hnsw
    search = search_hnsw
    data = None; indices = None; indptr = None; index = None; num_query = None; num_base = None; total_base = None
    keep_data = None; remap = None; remap_inv = None
    train_time = None; search_time = None; model_size = None
    
    def __init__(self, hp):
        for k, v in hp.items():
            self.hp[k] = v
        self.hp['model_file'] = '%s/hnsw_%s_%d_%d.bin'%(results_dir, self.hp['name'], self.hp['efC'], self.hp['M'])
        self.total_base = self.hp['data'].shape[0]
        self.preprocess()
        
    def preprocess(self):
        if self.hp['metric'] == 'cosinesimil':
            norms = np.linalg.norm(self.hp['data'], axis=1)
            self.keep_data = np.where(abs(norms-1) < 0.05)[0]
            del norms
        else:
            self.keep_data = np.arange(self.total_base)
        print('Keeping %d/%d base points after preprocess'%(self.keep_data.shape[0], self.total_base))
        
        self.remap = np.vectorize({i : v for i, v in enumerate(self.keep_data)}.get)
        self.remap_inv = np.vectorize({v : i for i, v in enumerate(self.keep_data)}.get)
        self.hp['data'] = self.hp['data'][self.keep_data]
        
    def getnns(self):
        score_mat = csr_matrix((self.data.ravel().astype(float), self.remap(self.indices.ravel().astype(int)), self.indptr), (self.hp['query'].shape[0], self.total_base))
#         del self.data, self.indptr, self.indices
        return score_mat
    
    def load_index(self, filename = None):
        if filename is None: filename = self.hp['model_file']
        self.index = nmslib.init(method='hnsw', space=self.hp['metric'])
        nmslib.loadIndex(self.index, filename)

