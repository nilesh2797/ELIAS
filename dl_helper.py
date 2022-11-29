import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp

from typing import Dict

def unwrap(net):
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        return net.module
    return net

def ToD(batch, device):
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    if isinstance(batch, Dict):
        for outkey in batch:
            batch[outkey] = ToD(batch[outkey], device)
    return batch

def get_index_values(spmat, row_index, add_one=False):
    start = spmat.indptr[row_index]; end = spmat.indptr[row_index+1]
    row_data = spmat.data[start:end]
    row_indices = spmat.indices[start:end]
    return row_indices, row_data

def dedup_long_tensor(x: torch.LongTensor, replace_val=0, stable=True):
    y, indices = x.sort(dim=-1, stable=True) # sort input x so that equal elements are together
    mask = ((y[:, 1:] - y[:, :-1]) != 0)
    y[:, 1:] = torch.where(mask, y[:, 1:], torch.full_like(y[:, 1:], replace_val))
    if stable: return y.scatter_(1, indices, y.clone()) # undo sort
    else: return y

def csr_to_bow_tensor(spmat):
    return {'inputs': torch.LongTensor(spmat.indices),
            'offsets': torch.LongTensor(spmat.indptr),
            'per_sample_weights': torch.Tensor(spmat.data)}

def csr_to_pad_tensor(spmat, pad):
    inds_tensor = torch.LongTensor(spmat.indices)
    data_tensor = torch.FloatTensor(spmat.data)
    return {'inds': torch.nn.utils.rnn.pad_sequence([inds_tensor[spmat.indptr[i]:spmat.indptr[i+1]] for i in range(spmat.shape[0])], batch_first=True, padding_value=pad),
           'vals': torch.nn.utils.rnn.pad_sequence([data_tensor[spmat.indptr[i]:spmat.indptr[i+1]] for i in range(spmat.shape[0])], batch_first=True, padding_value=0.0)}

def bert_fts_batch_to_tensor(input_ids, attention_mask):
    maxlen = attention_mask.sum(axis=1).max()
    return {'input_ids': torch.LongTensor(input_ids[:, :maxlen]), 
            'attention_mask': torch.LongTensor(attention_mask[:, :maxlen])}
    
def bow_fts_batch_to_tensor(batch):
    xlen = sum([len(b['inds']) for b in batch])
    ret = {'inputs': torch.zeros(xlen).long(), 
           'offsets': torch.zeros(len(batch)+1).long(),
           'per_sample_weights': torch.zeros(xlen)}
    offset = 0
    for i, b in enumerate(batch):
        new_offset = offset+len(b['inds'])
        ret['inputs'][offset:new_offset] = torch.Tensor(b['inds']).long()
        ret['per_sample_weights'][offset:new_offset] = torch.Tensor(b['vals'])
        ret['offsets'][i+1] = new_offset            
        offset = new_offset
    return ret

class BatchIterator():
    def __init__(self, dataset, iter_bsz=256):
        self.dataset = dataset
        self.iter_bsz = iter_bsz
        self.iter_idx = 0

    def __call__(self, iter_bsz):
        self.iter_bsz = iter_bsz if iter_bsz is not None else self.iter_bsz
        return self

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.ceil(len(self.dataset)/self.iter_bsz))

    def __next__(self):
        if self.iter_idx*self.iter_bsz > len(self.dataset):
            self.iter_idx = 0
            raise StopIteration
        self.iter_idx += 1
        ids = np.arange(self.iter_bsz*(self.iter_idx-1), min(len(self.dataset), self.iter_bsz*self.iter_idx))
        return {'xfts': self.dataset.get_fts(ids), 'ids': torch.LongTensor(ids)}

def apply_and_accumulate(data_loader, func, accelerator, display_name='Iterating', **kwargs):
    is_main_proc = accelerator is None or accelerator.is_main_process
    out = {}
    with torch.no_grad():
        for b in tqdm(data_loader, leave=True, desc=display_name, disable=not is_main_proc):
            b_out = func(b, **kwargs)

            if accelerator is not None:
                b_out = accelerator.gather(b_out)
                b['ids'] = accelerator.gather(b['ids'])

            if is_main_proc:
                for k in b_out.keys():
                    if k not in out:
                        out[k] = np.zeros((len(data_loader.dataset), b_out[k].shape[-1]))
                    out[k][b['ids'].cpu()] = b_out[k].detach().cpu().numpy()
            del b_out, b
    return out

def create_tf_pooler(pooler_type: str):
    if pooler_type == 'seq-clf':
        def f(tf_output, batch_data):
            return tf_output.logits
        return f, 768
    elif pooler_type == 'pooler':
        def f(tf_output, batch_data):
            return tf_output['pooler_output']
        return f, 768
    elif pooler_type == 'cls':
        def f(tf_output, batch_data):
            return tf_output['last_hidden_state'][:, 0]
        return f, 768
    elif pooler_type == 'lightxml':
        def f(tf_output, batch_data):
            h = tf_output['hidden_states']
            return torch.hstack([h[-i-1][:, 0] for i in range(5)])
        return f, 768*5
    elif pooler_type == 'mean':
        def f(tf_output, batch_data):
            last_hidden_state = tf_output['last_hidden_state']
            input_mask_expanded = batch_data['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_hidden_state / sum_mask
        return f, 768
    else:
        print(f'Unknown pooler type encountered: {pooler_type}, using identity pooler instead')
        def f(tf_output, batch_data):
            return tf_output
        return f, 768

class SparseLinear(nn.Module):
    def __init__(self, numx, numy):
        super(SparseLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand((numx, numy)))
        self.bias = nn.Parameter(torch.rand((numx,)))
        self.reset()
        
    def reset(self):
        nn.init.kaiming_uniform_(self.weight)
        
    def forward(self, embs, shortlist):
        weight_reshaped = self.weight[shortlist.reshape(-1)].reshape(embs.shape[0], -1, embs.shape[1]).permute(0, 2, 1)
        bias_reshaped = self.bias[shortlist]
        out = torch.bmm(embs.unsqueeze(1), weight_reshaped).squeeze() + bias_reshaped
        return out