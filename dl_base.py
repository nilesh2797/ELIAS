import os, sys, math, random, time
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer
from collections import OrderedDict
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import tqdm, trange
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable
import logging
logger = logging.getLogger(__name__)

import transformers
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, RobertaModel


def ToD(batch, device):
    if isinstance(batch, torch.Tensor):
        batch = batch.to(device)
    if isinstance(batch, Dict):
        for outkey in batch:
            if isinstance(batch[outkey], torch.Tensor):
                batch[outkey] = batch[outkey].to(device)
            if isinstance(batch[outkey], Dict):
                for inkey in batch[outkey]:
                    if isinstance(batch[outkey][inkey], torch.Tensor):
                        batch[outkey][inkey] = batch[outkey][inkey].to(device)
    return batch

def np_relu(x):
    return np.maximum(0, x)

def get_index_values(spmat, row_index, add_one=False):
    start = spmat.indptr[row_index]; end = spmat.indptr[row_index+1]
    row_data = spmat.data[start:end]
    row_indices = spmat.indices[start:end]
    
    if(add_one):
        row_indices = row_indices + 1

    return row_indices, row_data

def dedup_tensor(x: torch.LongTensor, replace_val=0, return_indices=False):
    # Assumption: x is a 2D torch.LongTensor
    offset = 1 - x.min()
    y, indices = x.sort(dim=-1, stable=True)
    y += offset # make min of y = 1
    mask = ((y[:, 1:] - y[:, :-1]) != 0)
    y[:, 1:] = torch.where(mask, y[:, 1:], torch.full_like(y[:, 1:], replace_val+offset))
#     y[:, 1:] *= mask
#     y[:, 1:] += (~mask)*torch.full_like(y[:, 1:], replace_val+offset)
    y -= offset
    return (y, indices) if return_indices else y

def csr_to_bow_tensor(spmat):
    return {'inputs': torch.LongTensor(spmat.indices),
            'offsets': torch.LongTensor(spmat.indptr),
            'per_sample_weights': torch.Tensor(spmat.data)}

def csr_to_pad_tensor(spmat, pad):
    inds_tensor = torch.LongTensor(spmat.indices)
    data_tensor = torch.FloatTensor(spmat.data)
    return {'inds': torch.nn.utils.rnn.pad_sequence([inds_tensor[spmat.indptr[i]:spmat.indptr[i+1]] for i in range(spmat.shape[0])], batch_first=True, padding_value=pad),
           'vals': torch.nn.utils.rnn.pad_sequence([data_tensor[spmat.indptr[i]:spmat.indptr[i+1]] for i in range(spmat.shape[0])], batch_first=True, padding_value=0.0)}

def old_csr_to_pad_tensor(spmat, pad):
    maxlen = spmat.getnnz(1).max()
    ret = {'inds': torch.full((spmat.shape[0], maxlen), pad).long().flatten(),
           'vals': torch.zeros(spmat.shape[0], maxlen).flatten()}
    ptrs = []
    for i in range(spmat.shape[0]):
        ptrs.append(torch.arange(i*maxlen, i*maxlen + spmat.indptr[i+1] - spmat.indptr[i]))
    ptrs = torch.cat(ptrs)
    ret['inds'][ptrs] = torch.LongTensor(spmat.indices)
    ret['inds'] = ret['inds'].reshape((spmat.shape[0], maxlen))
    ret['vals'][ptrs] = torch.Tensor(spmat.data)
    ret['vals'] = ret['vals'].reshape((spmat.shape[0], maxlen))
    return ret

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

class FixedDataset(torch.utils.data.Dataset):

    def __init__(self, point_features, labels, label_features=None, shorty=None):
        self.point_features = point_features
        self.label_features = label_features
        self.labels = labels
        self.shorty = shorty

        print("------ Some stats about the dataset ------")
        print("Shape of X_Xf       : ", self.point_features.shape)
        print("Shape of X_Y        : ",  self.labels.shape, end='\n\n')
        print("Avg. lbls per point : %.2f"%(np.average(np.array(self.labels.sum(axis=1)))))
        print("Avg. fts per point  : %.2f"%(np.average(self.point_features.astype(np.bool).sum(axis=1))))
        print("------------------------------------------")

        self.num_Y = self.labels.shape[1]
        self.num_Xf = self.point_features.shape[1]

    def __getitem__(self, index):
        ret = {'index': index, 'xfts': None, 'y': None, 'shorty': None}
        ret['xfts'] = self.get_fts(index, 'point')
        temp = get_index_values(self.labels, index)
        ret['y'] = {'inds': temp[0], 'vals': temp[1]}
        
        if not self.shorty is None:
            temp = get_index_values(self.shorty, index)
            ret['shorty'] = {'inds': temp[0], 'vals': temp[1]}
        return ret
    
    def get_fts(self, index, source='point'):
        if isinstance(self.point_features, sp.csr_matrix):
            if isinstance(index, int) or isinstance(index, np.int32):
                if source == 'label':
                    temp = get_index_values(self.label_features, index)
                else:
                    temp = get_index_values(self.point_features, index)
                return {'inds': temp[0], 
                        'vals': temp[1]}
            else:
                if source == 'label':
                    return csr_to_bow_tensor(self.label_features[index])
                else:
                    return csr_to_bow_tensor(self.point_features[index])
        else:
            if isinstance(index, int) or isinstance(index, np.int32):
                if source == 'label':
                    return self.label_features[index]
                else:
                    return self.point_features[index]
            else:
                if source == 'label':
                    return torch.Tensor(self.label_features[index])
                else:
                    return torch.Tensor(self.point_features[index])

    @property
    def num_instances(self):
        return self.point_features.shape[0]

    @property
    def num_labels(self):
        return self.labels.shape[1]
    
    def __len__(self):
        return self.point_features.shape[0]

class PreTokBertDataset(torch.utils.data.Dataset):
    def __init__(self, tokenization_folder, X_Y, max_len, token_type='bert-base-uncased', shorty=None, doc_type='trn', sample=None, iter_mode='pointwise'):
        self.max_len = max_len
        self.iter_mode = iter_mode
        self.sample = np.arange(X_Y.shape[0]) if sample is None else sample
        self.labels = X_Y[self.sample]
        self.shorty = shorty
        self.tokenizer = AutoTokenizer.from_pretrained(token_type)
        
        if not os.path.exists(tokenization_folder):
            print(f'Pre-Tokenized folder ({tokenization_folder}) not found')
            print(f'Help: python create_tokenized_files.py --data-dir Datasets/<dataset> --max-length {max_len}')
            sys.exit()
        
        self.X_ii = np.memmap(f"{tokenization_folder}/{doc_type}_doc_input_ids.dat", 
                             mode='r', shape=(X_Y.shape[0], max_len), dtype=np.int64)
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret
    
    def get_fts(self, indices, source='point'):
        X_ii = np.array(self.X_ii[self.sample[indices]])
        X_am = (X_ii != self.tokenizer.pad_token_id)
        return bert_fts_batch_to_tensor(X_ii, X_am)
   
    def __len__(self):
        return len(self.sample)
    
class OnlineBertDataset(torch.utils.data.Dataset):
    def __init__(self, X, X_Y, max_len, token_type='bert-base-uncased', shorty=None, sample=None, iter_mode='pointwise'):
        self.max_len = max_len
        self.iter_mode = iter_mode
        self.sample = np.arange(X_Y.shape[0]) if sample is None else sample
        self.labels = X_Y[self.sample]
        self.shorty = shorty
        self.X = np.array(X, dtype=object)
        self.tokenizer = AutoTokenizer.from_pretrained(token_type)
            
    def __getitem__(self, index):
        ret = {'index': index}
        return ret
    
    def get_fts(self, indices, source='point'):
        return self.tokenizer.batch_encode_plus(list(self.X[self.sample[indices]]), max_length=self.max_len, padding=True, truncation=True, return_tensors='pt', return_token_type_ids=False)
   
    def __len__(self):
        return len(self.sample)

class XCCollator():
    def __init__(self, numy, dataset, yfull=False):
        self.numy = numy
        self.yfull = yfull
        self.dataset = dataset
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = np.array([b['index'] for b in batch])
        
        batch_data = {'batch_size': torch.LongTensor([batch_size]),
                      'numy': torch.LongTensor([self.numy]),
                      'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
                      'ids': torch.LongTensor(ids),
                      'xfts': self.dataset.get_fts(ids, 'point')
                     }
            
        if self.dataset.shorty is not None:
            batch_data['shorty'] = csr_to_pad_tensor(self.dataset.shorty[ids], self.numy)
            
        if self.yfull:
            batch_data['yfull'] = torch.zeros(batch_size, self.numy+1).scatter_(1, batch_data['y']['inds'], batch_data['y']['vals'])[:, :-1]
                
        return batch_data

class TransformerInputLayer(nn.Module):
    def __init__(self, transformer, pooler_type='pooler'):
        super(TransformerInputLayer, self).__init__()
        self.transformer = transformer
        self.pooler = self.create_pooler(pooler_type)

    def forward(self, data):
        return self.pooler(self.transformer(**data, output_hidden_states=True), data)
    
    def create_pooler(self, pooler_type: str):
        if pooler_type == 'seq-clf':
            def f(tf_output, batch_data):
                return tf_output.logits
            return f
        elif pooler_type == 'pooler':
            self.dims = 768
            def f(tf_output, batch_data):
                return tf_output['pooler_output']
            return f
        elif pooler_type == 'cls':
            self.dims = 768
            def f(tf_output, batch_data):
                return tf_output['last_hidden_state'][:, 0]
            return f
        elif pooler_type == 'lightxml':
            self.dims = 768*5
            def f(tf_output, batch_data):
                h = tf_output['hidden_states']
                return torch.hstack([h[-i-1][:, 0] for i in range(5)])
            return f
        elif pooler_type == 'mean':
            self.dims = 768
            def f(tf_output, batch_data):
                last_hidden_state = tf_output['last_hidden_state']
                input_mask_expanded = batch_data['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_hidden_state = torch.sum(last_hidden_state * input_mask_expanded, 1)

                sum_mask = input_mask_expanded.sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)

                return sum_hidden_state / sum_mask
            return f
        else:
            print(f'Unknown pooler type encountered: {pooler_type}, using identity pooler instead')
            def f(tf_output, batch_data):
                return tf_output
            return f