import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_helper import create_tf_pooler, ToD, csr_to_pad_tensor, dedup_long_tensor, BatchIterator, SparseLinear, apply_and_accumulate
from transformers import AutoModel
import numpy as np
import scipy.sparse as sp

class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()

    def ToD(self, batch):
        return ToD(batch, self.get_device())

    def get_device(self):
        if hasattr(self, 'device'):
            return self.device
        return list(self.parameters())[0].device

    def get_embs(self, data_source, bsz=256, accelerator=None):
        self.eval()
        if isinstance(data_source, torch.utils.data.Dataset): 
            data_source = BatchIterator(data_source, bsz)
        
        out = apply_and_accumulate(
            data_source, 
            lambda b: {'embs': self.encode(self.ToD(b))},
            accelerator,
            display_name='Embedding'
            )
        return out['embs'] if 'embs' in out else None

    def _predict_batch(self, b, K):
        b = ToD(b, self.get_device())
        out = self(b)
        if isinstance(out, torch.Tensor): # BxL shaped out
            top_vals, top_inds = torch.topk(out, K)
        elif isinstance(out, tuple) and len(out) == 2: # (logits, indices) shaped out 
            top_vals, temp_inds = torch.topk(out[0], K)
            top_inds = torch.gather(out[1], 1, temp_inds)
        return {'top_vals': top_vals, 'top_inds': top_inds}

    def predict(self, data_source, K=100, bsz=256, accelerator=None):
        self.eval()
        if isinstance(data_source, torch.utils.data.Dataset):
            data_source = BatchIterator(data_source, bsz)

        out = apply_and_accumulate(
            data_source, 
            self._predict_batch,
            accelerator,
            display_name='Predicting',
            **{'K': K}
            )

        if accelerator is None or accelerator.is_main_process:
            labels = data_source.dataset.labels
            indptr = np.arange(0, labels.shape[0]*K+1, K)
            score_mat = sp.csr_matrix((out['top_vals'].ravel(), out['top_inds'].ravel(), indptr), labels.shape)
            # remove padding if any
            if any(score_mat.indices == labels.shape[1]):
                score_mat.data[score_mat.indices == labels.shape[1]] = 0
                score_mat.eliminate_zeros()
            return score_mat

    def update_non_parameters(self, *args, **kwargs):
        pass

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        loaded_state = torch.load(path, map_location='cpu')
        return self.load_state_dict(loaded_state, strict=True)

class SWANet(BaseNet):
    def __init__(self, args):
        super().__init__()
        self.swa_state = None
        self.use_swa = args.use_swa
        self.swa_start = args.swa_start
        self.swa_step_size = args.swa_step
        self.no_swa = lambda x : False

    def predict(self, *args, **kwargs):
        self.swa_swap_params()
        ret = super().predict(*args, **kwargs)
        self.swa_swap_params()
        return ret

    def get_embs(self, *args, **kwargs):
        self.swa_swap_params()
        ret = super().get_embs(*args, **kwargs)
        self.swa_swap_params()
        return ret

    def save(self, *args, **kwargs):
        self.swa_swap_params()
        super().save(*args, **kwargs)
        self.swa_swap_params()

    def update_non_parameters(self, epoch, step, *args, **kwargs):
        super().update_non_parameters(epoch, step, *args, **kwargs)
        if self.use_swa and epoch == self.swa_start and (not self.swa_state):
            self.swa_init()
        if step%self.swa_step_size == 0 and self.swa_state:
            self.swa_step()

    def swa_init(self):
        self.swa = True
        print('SWA initializing')
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            if not self.no_swa(n):
                self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if not self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                if not self.no_swa(n):
                    self.swa_state[n].mul_(1.0 - beta).add_(p.data.cpu(), alpha=beta)

    def swa_swap_params(self):
        if not self.swa_state:
            return
        device = self.get_device()
        for n, p in self.named_parameters():
            if not self.no_swa(n):
                self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
                self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].to(device)
        if hasattr(self, 'update'): self.update()

class TFEncoder(SWANet):
    def __init__(self, args):
        super().__init__(args)
        tf_args = {'add_pooling_layer': False} if args.tf.startswith('bert-base') else {} 
        self.tf = AutoModel.from_pretrained(args.tf, **tf_args) if args.tf else None
        self.tf_pooler, self.tf_dims = create_tf_pooler(args.tf_pooler)
        self.bottleneck = nn.Linear(self.tf_dims, args.bottleneck_dim) if args.bottleneck_dim else None
        self.embs_dim = args.embs_dim = args.bottleneck_dim if args.bottleneck_dim else self.tf_dims
        self.dropout = nn.Dropout(args.dropout)
        self.norm_embs = args.norm_embs
        self.amp_encode = args.amp_encode
      
    def encode(self, b):
        with torch.cuda.amp.autocast(self.amp_encode):
            embs = b['xfts']
            if self.tf is not None:
                embs = self.tf_pooler(self.tf(**embs, output_hidden_states=True), embs)
            if self.bottleneck is not None:
                embs = self.bottleneck(embs)

            embs = self.dropout(embs)
            if self.norm_embs:
                embs = F.normalize(embs)
            return embs.float()

class OvANet(TFEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.w = nn.Linear(args.embs_dim, args.numy)
        self.loss_with_logits = args.loss_with_logits
    
    def forward(self, b, activey = None):
        embs = self.encode(b)
        if activey is None: 
            out = self.w(embs)
        else: 
            out = F.linear(embs, self.w.weight[activey], self.w.bias[activey].view(-1))
            
        if not self.loss_with_logits:
            out = torch.sigmoid(out)
        return out

class ELIAS1(TFEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.numy = args.numy
        self.beam_size = args.beam_size
        self.max_num_trn_clusters = self.beam_size+5
        self.alpha = args.alpha
        self.K = args.K
        self.max_leaf = args.max_leaf
        self.C = args.C = int(2**np.ceil(np.log2(args.numy/args.max_leaf))) # Number of clusters
        self.wl_dim = args.wl_dim if args.wl_dim > 0 else args.embs_dim

        self.WC = nn.Linear(self.embs_dim, self.C) # Cluster classifier matrix
        self.WL_transform = nn.Linear(self.embs_dim, self.wl_dim) if (self.embs_dim != self.wl_dim) else nn.Identity() # Embeddings transform before applying label classifiers
        self.WL = SparseLinear(args.numy+1, self.wl_dim) # Label classifier matrix

        self.load_adjacency_matrix(args.A_init_path)
        
    def load_adjacency_matrix(self, fname):
        A = sp.load_npz(fname)
        self.register_buffer('parent', torch.tensor(np.concatenate([np.asarray(A.argmax(axis=0)).squeeze(), [self.C]])))
        A_nz_dict = csr_to_pad_tensor(A, self.numy) # Non-zero indices and values in cluster-label adjacency matrix
        self.register_buffer('A_nz_inds', A_nz_dict['inds'])
        self.register_buffer('A_nz_vals', A_nz_dict['vals'])
            
    def get_cluster_shortlist(self, cluster_scores, b=None):
        cluster_shortlist_vals, cluster_shortlist_inds = torch.topk(cluster_scores, self.beam_size, dim=1)

        # Teacher forcing: add clusters belonging to positive labels
        if self.training:
            pos_batch_cluster_inds = self.parent[b['y']['inds']]
            cluster_shortlist_inds = torch.hstack((pos_batch_cluster_inds, cluster_shortlist_inds))
            cluster_shortlist_vals = torch.hstack((b['y']['vals'], cluster_shortlist_vals))
            # Deduplicate cluster indices
            cluster_shortlist_inds = dedup_long_tensor(cluster_shortlist_inds, self.C) # dedup indices of shortlisted clusters
            pad_mask = (cluster_shortlist_inds == self.C) # figure out where duplicate entries were
            cluster_shortlist_inds[pad_mask] = torch.randint(0, self.C, (pad_mask.sum(), ), device=pad_mask.device) # fill random cluster indices at duplicate entries
            cluster_shortlist_vals = cluster_scores.gather(1, cluster_shortlist_inds) # re-compute shortlist cluster values
            # If there are too many shortlisted clusters then sample based on the cluster scores
            if cluster_shortlist_inds.shape[1] > self.max_num_trn_clusters:
                sampled_inds = torch.multinomial(cluster_shortlist_vals, self.max_num_trn_clusters, replacement=False)
                cluster_shortlist_inds = cluster_shortlist_inds.gather(1, sampled_inds)
                cluster_shortlist_vals = cluster_shortlist_vals.gather(1, sampled_inds)

        return cluster_shortlist_inds, cluster_shortlist_vals

    def expand_cluster_shortlist(self, cluster_shortlist_inds, cluster_shortlist_vals):
        bsz = cluster_shortlist_inds.shape[0]
        label_shortlist_inds = self.A_nz_inds[cluster_shortlist_inds].reshape(bsz, -1) # N x (B*M)
        label_shortlist_vals = torch.einsum('nb,nbm->nbm', cluster_shortlist_vals, self.A_nz_vals[cluster_shortlist_inds]).reshape(bsz, -1) # N x (B*M)
        return label_shortlist_inds, label_shortlist_vals

    def forward(self, b):
        b = self.ToD(b)
        embs = self.encode(b)
        
        cluster_scores = torch.clamp(self.alpha*F.softmax(self.WC(embs), dim=1), min=0, max=1)
        cluster_shortlist_inds, cluster_shortlist_vals = self.get_cluster_shortlist(cluster_scores, b)

        label_shortlist_inds, label_shortlist_vals = self.expand_cluster_shortlist(cluster_shortlist_inds, cluster_shortlist_vals)

        topK_label_shortlist_vals, sorted_inds = label_shortlist_vals.topk(min(self.K, label_shortlist_vals.shape[-1]))
        topK_label_inds = label_shortlist_inds.gather(1, sorted_inds)
        
        topK_label_wl_vals = torch.sigmoid(self.WL(self.WL_transform(embs), topK_label_inds))
        topK_label_vals = topK_label_shortlist_vals * topK_label_wl_vals

        if self.training:
            return (topK_label_vals, topK_label_inds, label_shortlist_vals, label_shortlist_inds)
        else:
            return (topK_label_vals, topK_label_inds)

class ELIAS2(ELIAS1):
    def __init__(self, args):
        super().__init__(args)
        self.beta = args.beta
        self.no_swa = lambda n: (n == 'A_nz_vals_param')
        # Leranable parameter which determines the weights (A_nz_vals) in the cluster-label adjacency matrix A
        self.register_parameter('A_nz_vals_param', nn.Parameter(torch.rand_like(self.A_nz_vals)))
        self.update_A()

    def update_A_nz_vals(self):
        self.A_nz_vals = torch.clamp(self.beta*F.softmax(self.A_nz_vals_param, dim=-1), min=0, max=1)
                
    def update_A(self):    
        with torch.no_grad():
            self.update_A_nz_vals()
            parent_val = torch.zeros((self.numy+1, ), device=self.get_device())
            for c in range(self.A_nz_vals.shape[0]):
                cy = self.A_nz_inds[c]
                self.parent[cy] = torch.where(parent_val[cy] < self.A_nz_vals[c], torch.full_like(cy, c), self.parent[cy])
                parent_val[cy] = torch.max(parent_val[cy], self.A_nz_vals[c])

    def update_non_parameters(self, epoch, step, *args, **kwargs):
        super().update_non_parameters(epoch, step, *args, **kwargs)
        if not hasattr(self, 'last_update_epoch') or epoch != self.last_update_epoch:
            self.last_update_epoch = epoch
            self.update_A()

    def expand_cluster_shortlist(self, cluster_shortlist_inds, cluster_shortlist_vals):
        # Update A_nz_vals since model parameters change at every step during training
        if self.training: 
            self.update_A_nz_vals()
        label_shortlist_inds, label_shortlist_vals = super().expand_cluster_shortlist(cluster_shortlist_inds, cluster_shortlist_vals)

        # Arrange shortlisted labels in sorted order based on shortlist scores, this will ensure that after dedup only the index with maximum score remains among duplicate labels
        label_shortlist_vals, sort_inds = torch.sort(label_shortlist_vals, descending=True)
        label_shortlist_inds = label_shortlist_inds.gather(1, sort_inds)

        # Deduplicate labels in the shortlist, 
        label_shortlist_inds = dedup_long_tensor(label_shortlist_inds, self.numy)
        label_shortlist_vals[label_shortlist_inds == self.numy] = 0 # zero out values at duplicate label indices

        return label_shortlist_inds, label_shortlist_vals

    def load(self, path):
        loaded_state = torch.load(path, map_location='cpu')
        # Don't load WL* parameters if A was not learnable in the load checkpoint
        if not 'A_nz_vals_param' in loaded_state.keys():
            loaded_state = {k: v for k, v in loaded_state.items() if k.startswith("tf.") or k.startswith("WC.")}
        ret = self.load_state_dict(loaded_state, strict=False)
        self.update_A()
        return ret
        
    def retain_topk(self, K, clean=False):
        with torch.no_grad():
            self.A_nz_vals_param.data, topk_inds = self.A_nz_vals_param.topk(K)
            self.A_nz_inds = self.A_nz_inds.gather(1, topk_inds)
            if clean: self.A_nz_vals_param.data[:] = 1
            self.update_A_nz_vals()

NETS = {
    'ova-net': OvANet,
    'elias-1': ELIAS1,
    'elias-2': ELIAS2
    }
