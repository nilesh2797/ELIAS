import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_base import TransformerInputLayer, ToD, csr_to_pad_tensor, dedup_tensor
from transformers import AutoModel
import numpy as np
import scipy.sparse as sp
from xclib.utils.sparse import retain_topk
from tqdm import tqdm

from pecos.xmc import Indexer, LabelEmbeddingFactory
from sklearn.preprocessing import normalize

''' ------------------------------- Net ------------------------------- '''
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        tf_args = {'add_pooling_layer': False} if args.tf.startswith('bert-base') else {} 
        self.encoder = TransformerInputLayer(AutoModel.from_pretrained(args.tf, **tf_args), args.tf_pooler) if args.tf else None
        self.encoder_bottleneck = nn.Linear(self.encoder.dims, args.bottleneck) if args.bottleneck else None
        args.embs_dim = args.bottleneck if args.bottleneck else self.encoder.dims
        self.dropout = nn.Dropout(args.dropout)
        self.norm_embs = args.norm_embs
        self.amp_encode = args.amp_encode
        self.swa = False
        self.no_swa = lambda x : False
        
    def ToD(self, batch):
        return ToD(batch, self.get_device())
      
    def get_device(self):
        if hasattr(self, 'device'):
            return self.device
        return list(self.parameters())[0].device
    
    def get_embs(self, tst_dataset, bsz=256):
        self.eval()
        self.swa_swap_params()
        total = len(tst_dataset)
        embs = []
        with torch.no_grad():
            for ctr in tqdm(range(0, total, bsz), leave=True, desc='Encoding'):
                b = self.ToD({'xfts': tst_dataset.get_fts(range(ctr, min(ctr+bsz, total)))})
                embs.append(self.encode(b).detach().cpu().numpy())
                del b
        self.swa_swap_params()
        return np.vstack(embs)
      
    def encode(self, b):
        with torch.cuda.amp.autocast(self.amp_encode):
            embs = b['xfts']
            if self.encoder is not None:
                embs = self.encoder(embs)
            if self.encoder_bottleneck is not None:
                embs = self.encoder_bottleneck(embs)

            embs = self.dropout(embs)
            if self.norm_embs:
                embs = F.normalize(embs)
            return embs.float()
    
    def predict(self, tst_loader, K=100):
        tst_X_Y = tst_loader.dataset.labels
        data = np.zeros((tst_X_Y.shape[0], K))
        inds = np.zeros((tst_X_Y.shape[0], K)).astype(np.int32)
        indptr = np.arange(0, tst_X_Y.shape[0]*K+1, K)
        self.eval()
        self.swa_swap_params()

        with torch.no_grad():
            for b in tqdm(tst_loader, leave=True, desc='Evaluating'):
                b = ToD(b, self.get_device())
                out = self(b)
                if isinstance(out, torch.Tensor): # BxL shaped out
                    top_data, top_inds = torch.topk(out, K)
                elif isinstance(out, tuple) and len(out) == 2: # (logits, indices) shaped out 
                    top_data, temp_inds = torch.topk(out[0], K)
                    top_inds = torch.gather(out[1], 1, temp_inds)
                    del temp_inds
                else:
                    print(f'Got unsupported type output: {type(out)}: {out}')
                    break
                data[b['ids'].cpu()] = top_data.detach().cpu().numpy()
                inds[b['ids'].cpu()] = top_inds.detach().cpu().numpy()
                del top_data, top_inds, b, out

        # torch.cuda.empty_cache() 
        self.swa_swap_params()
        score_mat = sp.csr_matrix((data.ravel(), inds.ravel(), indptr), tst_X_Y.shape)
        
        # remove padding if any
        if any(score_mat.indices == tst_X_Y.shape[1]):
            score_mat.data[score_mat.indices == tst_X_Y.shape[1]] = 0
            score_mat.eliminate_zeros()
        return score_mat
    
    def save_model(self, path):
        self.swa_swap_params()
        torch.save(self.state_dict(), path)
        self.swa_swap_params()

    def swa_init(self):
        self.swa = True
        print('SWA initializing')
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            if not self.no_swa(n):
                self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if not self.swa: return
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                if not self.no_swa(n):
                    self.swa_state[n].mul_(1.0 - beta).add_(p.data.cpu(), alpha=beta)

    def swa_swap_params(self):
        if not self.swa: return
        if 'models_num' not in self.swa_state:
            return
        device = self.get_device()
        for n, p in self.named_parameters():
            if not self.no_swa(n):
                self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
                self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].to(device)
        if hasattr(self, 'update'): self.update()

''' ------------------------------- FCNet ------------------------------- '''
class FCNet(Net):
    def __init__(self, args):
        super(FCNet, self).__init__(args)
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
    
''' ------------------------------- ELIAS ------------------------------- '''
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
    
def gen_A_init(net, trn_embs, trn_Y_X, kappa, device='cpu', bsz=100000):
    trn_tensor_embs = torch.tensor(trn_embs).to(device)
    M_indices = []; M_values = []
    with torch.no_grad():
        for ctr in tqdm(range(0, trn_tensor_embs.shape[0], bsz)):
            res = torch.mm(trn_tensor_embs[ctr:ctr+bsz], net.w1.weight.T.to(device)) + net.w1.bias.reshape(1, -1).to(device)
            res = torch.clamp(net.alpha*F.softmax(res, dim=1), min=0, max=1)
            temp = res.topk(net.beam, dim=1)
            M_indices.append(temp.indices)
            M_values.append(temp.values)

    M_indices = torch.vstack(M_indices)
    M_values = torch.vstack(M_values)

    numx = M_indices.shape[0]
    K = M_indices.shape[1]
    data_bin = np.ones(numx * K) 
    data_val = M_values.reshape(-1).cpu().numpy()
    inds = M_indices.reshape(-1).cpu().numpy()
    indptr = np.arange(0, numx*K + 1, K)
    shape = (trn_embs.shape[0], net.w1.weight.shape[0])
    
    M_bin = sp.csr_matrix((data_bin, inds, indptr), shape)
    M_val = sp.csr_matrix((data_val, inds, indptr), shape)
    C_bin = trn_Y_X.dot(M_bin)
    C_val = trn_Y_X.dot(M_val)

    CT_bin = C_bin.T.tocsr()
    CT_val = C_val.T.tocsr()
    CT_bin_topk = retain_topk(CT_bin, k=kappa)
    CT_val_topk = retain_topk(CT_val, k=kappa)
    A_init = CT_bin_topk + CT_val_topk
    A_init.sum_duplicates()
#     A_init = retain_topk(A_init, k=kappa)

    return A_init

class ELIAS(Net):
    def __init__(self, args, rand_init=True):
        super(ELIAS, self).__init__(args)
    
        self.device = args.device
        self.stage = args.stage
        self.beam = args.beam
        self.max_trn_cluster = self.beam+5
        self.numy = args.numy
        self.loss_with_logits = args.loss_with_logits
        self.tau = 1
        self.alpha = args.alpha
        self.beta = int(args.max_leaf*args.beta_gain)
        self.K = args.K
        self.C = args.C
        self.max_leaf = args.max_leaf
        self.embs_dim = args.embs_dim
        self.clf_dim = args.clf_dim if args.clf_dim > 0 else args.embs_dim
        
        self.w1 = nn.Linear(self.embs_dim, args.C+1)
        self.transform = nn.Linear(self.embs_dim, self.clf_dim) if (self.embs_dim != self.clf_dim) else nn.Identity()
        self.w_leaf = SparseLinear(args.numy+1, self.clf_dim)

        self.bow_fts = normalize(sp.load_npz(f'{args.DATA_DIR}/X.trn.npz'), copy=False).astype(np.float32)
        self.OUT_DIR = args.OUT_DIR
        self.cmat_seed = args.cmat_seed
        self.cmat_update = args.cmat_update
        if os.path.exists(args.A_init_file):
            A_init = sp.load_npz(args.A_init_file)
            temp = csr_to_pad_tensor(A_init, args.numy)
            self.A_nz_inds = temp['inds'].to(args.device)
            if self.stage == 1:
                self.A_nz_vals = temp['vals'].to(args.device)
            else:
                # self.A_nz_vals = nn.Parameter(torch.rand(*self.A_nz_inds.shape))
                init_vals = temp['vals'].float() / torch.clamp(temp['vals'].float().max(dim=1).values.reshape(-1, 1), min=1e-8)
                init_vals += 0.2*torch.rand(*self.A_nz_inds.shape)
                self.A_nz_vals = nn.Parameter(init_vals)
            self.update()
        
        self.no_swa = lambda n: (n == 'A_nz_vals')
        
    def retain_topk(self, K, clean=False):
        with torch.no_grad():
            self.A_norm_nz_vals = torch.clamp(self.beta*F.softmax(self.A_nz_vals/self.tau, dim=-1), min=0, max=1)
            self.A_nz_vals.data, sorted_inds = self.A_nz_vals.topk(K)
            self.A_nz_inds = self.A_nz_inds.gather(1, sorted_inds)
            self.A_norm_nz_vals = self.A_norm_nz_vals.gather(1, sorted_inds)
            
            if clean:
                self.A_nz_vals.data[:] = 1
                self.A_norm_nz_vals[:] = 1
                
    def update(self, epoch=-1, trn_dataset=None, **kwargs):    
        if self.stage == 1 and self.cmat_update and (epoch == 0 or epoch == 5):
            print(f'Updating clusters at epoch {epoch}...')
            trn_embs = normalize(self.get_embs(trn_dataset), copy=False).astype(np.float32)
            cluster_embs = sp.hstack((self.bow_fts, trn_embs)).tocsr()

            label_feat = LabelEmbeddingFactory.create(trn_dataset.labels, cluster_embs, method="pifa")
            cmat = Indexer.gen(label_feat, 
                               indexer_type="hierarchicalkmeans", 
                               nr_splits=self.C, 
                               max_leaf_size=self.max_leaf, 
                               seed=self.cmat_seed).chain[-1].T
            cmat.resize(self.C+1, self.numy+1)
            sp.save_npz(f'{self.OUT_DIR}/cmat.npz', cmat)
            temp = csr_to_pad_tensor(cmat, self.numy)
            self.A_nz_inds = temp['inds'].to(self.get_device())
            self.A_nz_vals = temp['vals'].to(self.get_device())

        with torch.no_grad():
            self.A_norm_nz_vals = torch.clamp(self.beta*F.softmax(self.A_nz_vals/self.tau, dim=-1), min=0, max=1).to(self.get_device())
            self.parent = torch.zeros((self.numy+1, ), dtype=torch.long, device=self.get_device())
            parent_val = torch.zeros((self.numy+1, ), device=self.get_device())
            for c in range(self.A_norm_nz_vals.shape[0]):
                cy = self.A_nz_inds[c]
                self.parent[cy] = torch.where(parent_val[cy] < self.A_norm_nz_vals[c], torch.full(cy.shape, c, device=self.get_device()), self.parent[cy])
                parent_val[cy] = torch.max(parent_val[cy], self.A_norm_nz_vals[c])

    def forward(self, b, K=0):
        b = self.ToD(b)
        embs = self.encode(b)
        bsz = embs.shape[0]
        K = max(self.beam, K)
        
        out1 = torch.clamp(self.alpha*F.softmax(self.w1(embs), dim=1), min=0, max=1)[:, :-1]
        out1_topk_vals, out1_topk_inds = torch.topk(out1, K, dim=1)
    
        if self.training:
            if self.A_nz_vals.requires_grad:
                self.A_norm_nz_vals = torch.clamp(self.beta*F.softmax(self.A_nz_vals/self.tau, dim=-1), min=0, max=1)
            if self.w1.weight.requires_grad:
                # add parent clusters in shortlisted clusters
                out1_pos_inds = self.parent[b['y']['inds']]
                out1_topk_inds = torch.hstack((out1_topk_inds, out1_pos_inds))
                out1_topk_vals = torch.hstack((out1_topk_vals, b['y']['vals']))

                out1_topk_inds, rearranged_inds = dedup_tensor(out1_topk_inds, replace_val=self.C, return_indices=True)
                out1_topk_vals = out1_topk_vals.gather(1, rearranged_inds)
                if self.max_trn_cluster > 0 and self.max_trn_cluster < out1_topk_vals.shape[1]:
                    sampled_inds = torch.multinomial(out1_topk_vals, self.max_trn_cluster)
                    out1_topk_inds = out1_topk_inds.gather(1, sampled_inds)
                pad_mask = (out1_topk_inds == self.C)
                out1_topk_inds[pad_mask] = torch.randint(0, self.C, (pad_mask.sum(), ), device=pad_mask.device)
        out1_topk_vals = out1.gather(1, out1_topk_inds)
        
        topk_C_vals = self.A_norm_nz_vals[out1_topk_inds.flatten()]
        topk_C_vals *= out1_topk_vals.reshape(-1, 1)
        topk_C_vals = topk_C_vals.reshape(bsz, -1)
        
        topk_C_inds = self.A_nz_inds[out1_topk_inds.flatten()].reshape(bsz, -1)
        topk_C_vals, sort_inds = torch.sort(topk_C_vals, descending=True)
        topk_C_inds = topk_C_inds.gather(1, sort_inds)
        
        topk_C_inds, rearranged_inds = dedup_tensor(topk_C_inds, self.numy, return_indices=True)
        topk_C_vals = topk_C_vals.gather(1, rearranged_inds)
        topk_C_vals[topk_C_inds == self.numy] = 0 # handle pad index
        
        shorty_vals, sorted_inds = topk_C_vals.topk(min(self.K, topk_C_vals.shape[-1]))
        shorty_inds = topk_C_inds.gather(1, sorted_inds)
                
        out2 = self.w_leaf(self.transform(embs), shorty_inds)
        out2 = out2 if self.loss_with_logits else torch.sigmoid(out2)
        out = (out2 + shorty_vals) if self.loss_with_logits else (out2 * shorty_vals)

        return (out, shorty_inds, topk_C_vals, topk_C_inds) if self.training else (out, shorty_inds)
    
    def old_forward(self, b):
        b = self.ToD(b)
        embs = self.encode(b)
        bsz = embs.shape[0]

        out1 = torch.clamp(self.alpha*F.softmax(self.w1(embs), dim=1), min=0, max=1)
        out1_topk_vals, out1_topk_inds = torch.topk(out1, self.beam, dim=1)
    
        if self.training:
            if self.A_nz_vals.requires_grad:
                self.A_norm_nz_vals = torch.clamp(self.beta*F.softmax(self.A_nz_vals/self.tau, dim=-1), min=0, max=1)
            if self.w1.weight.requires_grad:
                # add parent clusters in shortlisted clusters
                out1_pos_inds = self.parent[b['y']['inds']]
                out1_topk_inds = torch.hstack((out1_topk_inds, out1_pos_inds))
                out1_topk_vals = torch.hstack((out1_topk_vals, b['y']['vals']))

                out1_topk_inds, rearranged_inds = dedup_tensor(out1_topk_inds, replace_val=self.C, return_indices=True)
                out1_topk_vals = out1_topk_vals.gather(1, rearranged_inds)
                if self.max_trn_cluster > 0 and self.max_trn_cluster < out1_topk_vals.shape[1]:
                    sampled_inds = torch.multinomial(out1_topk_vals, self.max_trn_cluster)
                    out1_topk_inds = out1_topk_inds.gather(1, sampled_inds)
                pad_mask = (out1_topk_inds == self.C)
                out1_topk_inds[pad_mask] = torch.randint(0, self.C, (pad_mask.sum(), ), device=pad_mask.device)
        out1_topk_vals = out1.gather(1, out1_topk_inds)
        
        EPSILON = 1e-8
        num_trn_cluster = out1_topk_inds.shape[1]
        topk_C_vals = (self.A_norm_nz_vals[out1_topk_inds.flatten()] * out1_topk_vals.reshape(-1, 1)).reshape(bsz, num_trn_cluster, -1)
        topk_C_vals -= torch.min(0.99*topk_C_vals, 10000*EPSILON*(1-torch.rand_like(topk_C_vals))) # add noise
        topk_C_inds = self.A_nz_inds[out1_topk_inds.flatten()].reshape(bsz, num_trn_cluster, -1)
        
        buf = torch.zeros((bsz, self.numy+1), device=out1.device)
        with torch.no_grad():
            for k in range(num_trn_cluster):
                buf.scatter_(1, topk_C_inds[:, k, :], torch.max(buf.gather(1, topk_C_inds[:, k, :]), topk_C_vals[:, k, :]))
            
        topk_C_inds = topk_C_inds.reshape(bsz, -1)
        topk_C_vals = topk_C_vals.reshape(bsz, -1)
        anti_mask = abs(buf.gather(1, topk_C_inds) - topk_C_vals) > EPSILON
        topk_C_inds[anti_mask] = self.numy
        topk_C_vals[anti_mask] = 0
        del buf

        shorty_vals, sorted_inds = topk_C_vals.topk(min(self.K, topk_C_vals.shape[-1]))
        shorty_inds = topk_C_inds.gather(1, sorted_inds)

        out2 = self.w_leaf(self.transform(embs), shorty_inds)
        out2 = out2 if self.loss_with_logits else torch.sigmoid(out2)
        out = (out2 + shorty_vals) if self.loss_with_logits else (out2 * shorty_vals)

        return (out, shorty_inds, topk_C_vals, topk_C_inds) if self.training else (out, shorty_inds)

NETS = {
    'fcnet': FCNet,
    'elias': ELIAS
    }
