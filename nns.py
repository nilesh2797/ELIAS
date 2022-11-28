import numpy as np
import torch
import scipy.sparse as sp
import time
from tqdm import tqdm
import torch.nn.functional as F
if torch.__version__ > "1.11":
    torch.backends.cuda.matmul.allow_tf32 = True

class ExactSearch(object):
    def __init__(self, base, device, metric='ip', K=10, bsz=512, shard_size=1000000):
        self.base = torch.from_numpy(base) if isinstance(base, np.ndarray) else base
        self.device = device
        self.metric = metric
        self.K = K
        self.bsz = bsz
        self.shard_size = shard_size
        
        if self.metric == 'cosine':
            self.base = F.normalize(self.base, dim=1)
            self.metric = 'ip'
            
    def _to_csr(self, res):
        indptr = range(0, res['vals'].shape[0]*res['vals'].shape[1]+1, res['vals'].shape[1])
        score_mat = sp.csr_matrix((res['vals'].detach().cpu().numpy().ravel(), 
                                res['inds'].detach().cpu().numpy().ravel(), 
                                indptr), 
                               (res['vals'].shape[0], self.base.shape[0]))
        return score_mat
    
    def _agg_sharded_results(self, res, agg_bsz=100000):
        res['vals'] = torch.hstack(res['vals'])
        res['inds'] = torch.hstack(res['inds'])
        agg_res = {'inds': [], 'vals': []}
        nq = res['vals'].shape[0]
        
        with torch.no_grad():
            for i in tqdm(range(0, nq, agg_bsz), desc=f'Aggregating sharded results', leave=True):
                batch_vals = res['vals'][i:i+agg_bsz].to(self.device)
                batch_inds = res['inds'][i:i+agg_bsz].to(self.device)
                topk_vals, topk_inds = batch_vals.topk(k=self.K, sorted=True)
                agg_res['vals'].append(topk_vals.detach().cpu())
                agg_res['inds'].append(batch_inds.gather(1, topk_inds).detach().cpu())
        
        agg_res['vals'] = torch.vstack(agg_res['vals'])
        agg_res['inds'] = torch.vstack(agg_res['inds'])
        return agg_res
        
    def search(self, query):
        res = {'inds': [], 'vals': []}
        query = torch.from_numpy(query) if isinstance(query, np.ndarray) else query
        query = F.normalize(query) if self.metric == 'cosine' else query
        nq = query.shape[0]
        nb = self.base.shape[0]
        total_shards = (nb+self.shard_size-1)//self.shard_size
        print(f'Searching in {total_shards} shards')
        
        start = time.time()
        for shard_id, ctr in enumerate(range(0, nb, self.shard_size)):
            shard_res = {'inds': [], 'vals': []}
            shard = range(ctr, min(ctr+self.shard_size, nb))
            with torch.no_grad():
                base_shard_T = self.base[shard].T.to(self.device)

                for i in tqdm(range(0, nq, self.bsz), desc=f'Searching shard {shard_id+1}/{total_shards}', leave=True):
                    query_batch = query[i:i+self.bsz].to(self.device)
                    if self.metric == 'ip':
                        vals = torch.matmul(query_batch, base_shard_T)
                    elif self.metric.startswith('l_'):
                        p = int(self.metric.split('_')[-1])
                        vals = -torch.cdist(query_batch, base_shard_T.T, p=p)

                    topk_vals, topk_inds = torch.topk(vals, k=self.K, sorted=True)
                    shard_res['vals'].append(topk_vals.detach().cpu())
                    shard_res['inds'].append(topk_inds.detach().cpu())

            del base_shard_T, query_batch, vals, topk_vals, topk_inds
            shard_res['vals'] = torch.vstack(shard_res['vals'])
            shard_res['inds'] = torch.vstack(shard_res['inds'])
            res['vals'].append(shard_res['vals'])
            res['inds'].append(shard_res['inds']+ctr)
            
        if total_shards > 1:
            res = self._agg_sharded_results(res)
        else:
            res = {'vals': res['vals'][0], 'inds': res['inds'][0]}
        end = time.time()
        print('Total time, time per point : %.2fs, %.4f ms/pt'%(end-start, (end-start)*1000/nq))
            
        return self._to_csr(res)
