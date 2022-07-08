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
    
class BeamBCELoss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(BeamBCELoss, self).__init__()
        self.numy = args.numy
        self.gamma = args.beam_loss_gamma
        self.mask = torch.zeros(self.numy+1).long().to(args.device) # +1 to handle pad index
        if args.loss_with_logits:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, model, b):
        out1, out, shorty = model(b)
        
        yfull = torch.zeros((out.shape[0], self.numy+1), device=out.device).scatter_(1, b['y']['inds'], 1)
        yfull[:, -1] = 0.0
        targets  = torch.gather(yfull, 1, shorty)

        cluster_targets = torch.zeros_like(out1).scatter_(1, model.parent[b['y']['inds']], 1)
        cluster_targets[:, -1] = 0
        
        loss = self.criterion(out, targets) + self.gamma*self.criterion(out1, cluster_targets)
        del b, out, out1, yfull, targets, cluster_targets
        return loss
    
class JointLoss(nn.Module):
    def __init__(self, args, reduction='mean'):
        super(JointLoss, self).__init__()
        self.numy = args.numy
        self.mask = torch.zeros((256, self.numy+1), dtype=torch.bool).to(args.device) # +1 to handle pad index
        self.gamma = args.joint_loss_gamma
        if args.loss_with_logits:
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction=reduction)
        else:
            self.criterion = torch.nn.BCELoss(reduction=reduction)

    def forward(self, model, b):
        out, shorty, topk_C_vals, topk_C_inds = model(b)
        
        out_targets = torch.zeros_like(out)
        topk_C_targets = torch.zeros_like(topk_C_vals)
        mbsz = self.mask.shape[0]
        for i in range(0, out.shape[0], mbsz):
            self.mask.scatter_(1, b['y']['inds'][i:i+mbsz], True)
            self.mask[:, -1] = False
            out_targets[i:i+mbsz] = self.mask.gather(1, shorty[i:i+mbsz])
            topk_C_targets[i:i+mbsz] = self.mask.gather(1, topk_C_inds[i:i+mbsz])
            self.mask.scatter_(1, b['y']['inds'][i:i+mbsz], False)
        
        loss_precision = self.criterion(out, out_targets)
        
        targets, target_inds = topk_C_targets.topk(topk_C_targets.sum(dim=-1).max().long())
        topk_C_vals = topk_C_vals.gather(1, target_inds)
        topk_C_vals[targets < 1e-5] = 0
        loss_recall = self.criterion(topk_C_vals, targets)
        
        loss = loss_precision + self.gamma*loss_recall
        
        del b, out, targets, shorty, topk_C_vals, topk_C_inds, topk_C_targets, out_targets
        return loss
    
LOSSES = {
    'ova-bce': OvABCELoss,
    'batch-bce': BatchBCELoss,
    'beam-bce': BeamBCELoss,
    'joint': JointLoss
    }