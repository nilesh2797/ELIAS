import torch.nn as nn
import torch.nn.functional as F
import torch 

class BatchBCELoss(nn.Module):
    '''
        Assumes: model's forward function is able to take (b, batch_y)
    '''
    def __init__(self, args, reduction='mean'):
        super(BatchBCELoss, self).__init__()
        self.numy = args.numy
        self.mask = torch.zeros(self.numy+1).long().to(args.device) # +1 to handle pad index
        if args.loss_with_logits:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, model, b):
        batch_y = torch.unique(b['y']['inds'], sorted=True)
        self.mask[batch_y] = torch.arange(batch_y.shape[0], device=self.mask.device)
        batch_y_inds = self.mask[b['y']['inds']]
        
        valid_batch_y = batch_y 
        if batch_y[-1] == self.numy: # remove padded index if it is in batch_y
            valid_batch_y = batch_y[:-1]
        out = model(b, valid_batch_y)
        
        targets = torch.zeros((out.shape[0], batch_y.shape[0]), device=out.device).scatter_(1, batch_y_inds, 1)
        if batch_y[-1] == self.numy:
            targets = targets[:, :-1] # remove padded index if it is in batch_y
        self.mask[batch_y] = 0
        
        loss = self.criterion(out, targets)
        del b, out, targets, batch_y, batch_y_inds
        return loss
    
class OvABCELoss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(OvABCELoss, self).__init__()
        if args.loss_with_logits:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, model, b):
        out = model(b)
        targets = torch.zeros((out.shape[0], out.shape[1]+1), device=out.device).scatter_(1, b['y']['inds'], 1)[:, :-1]
        loss = self.criterion(out, targets)
        return loss
    
class ELIASLoss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super().__init__()
        self.numy = args.numy
        self.loss_lambda = args.loss_lambda
        self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, model, b):
        topK_label_vals, topK_label_inds, label_shortlist_vals, label_shortlist_inds = model(b)
        
        # Figure out ground truth target for topK labels and shortlisted labels
        with torch.no_grad():
            topK_label_targets = torch.zeros_like(topK_label_vals)
            label_shortlist_targets = torch.zeros_like(label_shortlist_vals)
            # Iterate over each column of b['y'] (groud truth) and check if the index matches or not
            for j in range( b['y']['inds'].shape[1]):
                inds_match_mask = (topK_label_inds ==  b['y']['inds'][:, [j]])
                topK_label_targets = torch.where(inds_match_mask,  b['y']['vals'][:, [j]], topK_label_targets)
                inds_match_mask = (label_shortlist_inds ==  b['y']['inds'][:, [j]]) & (label_shortlist_inds < self.numy)
                label_shortlist_targets = torch.where(inds_match_mask,  torch.tensor(1.0).to(inds_match_mask.device), label_shortlist_targets)

        loss_classification = self.criterion(topK_label_vals, topK_label_targets)
            
        # Figure out positively labelled shortlist targets since shortlist loss is only computed on positive entries
        pos_label_shortlist_targets, pos_label_shortlist_inds = label_shortlist_targets.topk(b['y']['inds'].shape[1])
        pos_label_shortlist_vals = label_shortlist_vals.gather(1, pos_label_shortlist_inds)
        pos_label_shortlist_vals[pos_label_shortlist_targets < 1e-5] = 0
        loss_shortlist = self.criterion(pos_label_shortlist_vals, pos_label_shortlist_targets)
        
        loss = loss_classification + self.loss_lambda*loss_shortlist
        
        del b, topK_label_vals, topK_label_inds, label_shortlist_vals, label_shortlist_inds, topK_label_targets, label_shortlist_targets
        return loss

class TripletOHNM(nn.Module):
    def __init__(self, args):
        super(TripletOHNM, self).__init__()
        self.numy = args.numy
        self.margin = args.loss_margin
        self.num_neg = args.loss_num_neg

    def forward(self, model, b):
        xembs = model.encode({'xfts': b['xfts']})
        yembs = model.encode({'xfts': b['yfts']})
        target = b['targets']
        sim = xembs @ yembs.T
        sim_p = torch.gather(sim, 1, b['pos-inds']).reshape(b['batch_size'], -1)
        neg_sim = torch.where(target < 1e-6, sim, torch.full_like(sim, -100))
        _, indices = torch.topk(neg_sim, largest=True, dim=1, k=self.num_neg)
        sim_n = sim.gather(1, indices)
        loss = torch.max(torch.zeros_like(sim_p), sim_n - sim_p + self.margin)
        mask = torch.where(loss != 0, torch.ones_like(loss), torch.zeros_like(loss))
        prob = torch.softmax(sim_n * mask, dim=1)
        reduced_loss =  (loss * prob).mean()
        
        del b, xembs, yembs, sim, neg_sim, sim_p, sim_n, mask, prob, loss
        return reduced_loss

class InfoNCE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.loss_tau = args.loss_tau
        self.loss_reduction = args.loss_reduction

    def forward(self, model, b):
        xembs = model.encode({'xfts': b['xfts']})
        yembs = model.encode({'xfts': b['yfts']})
        target = b['targets']
        sim = xembs @ yembs.T
        return F.cross_entropy(sim / self.loss_tau, target, reduction=self.loss_reduction)
    
LOSSES = {
    'ova-bce': OvABCELoss,
    'batch-bce': BatchBCELoss,
    'elias-loss': ELIASLoss,
    'triplet-ohnm': TripletOHNM,
    'infonce': InfoNCE
    }