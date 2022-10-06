import os, torch
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing
from transformers import AutoTokenizer

from resources import _c, get_inv_prop, load_filter_mat, _filter, compute_xmc_metrics
from dl_helper import unwrap, csr_to_bow_tensor, csr_to_pad_tensor, bert_fts_batch_to_tensor

class BaseDataset(torch.utils.data.Dataset):
	def __init__(self, labels: sp.csr_matrix, sample=None, filter_mat=None):
		super().__init__()
		self.sample = np.arange(labels.shape[0]) if sample is None else sample
		self.labels = labels[self.sample]
		self.filter_mat = filter_mat[self.sample] if filter_mat is not None else None

	def __getitem__(self, index):
		return {'index': index}

	def __len__(self):
		return len(self.sample)

class SimpleDataset(BaseDataset):
	def __init__(self, features, labels, **super_kwargs):
		super().__init__(labels, **super_kwargs)
		self.features = features
	
	def get_fts(self, indices):
		if isinstance(self.features, sp.csr_matrix):
			csr_to_bow_tensor(self.features[self.sample[indices]])
		else:
			torch.Tensor(self.features[self.sample[indices]])

class OfflineBertDataset(BaseDataset):
	def __init__(self, fname, labels, max_len, token_type='bert-base-uncased', **super_kwargs):
		super().__init__(labels, **super_kwargs)
		self.max_len = max_len
		self.tokenizer = AutoTokenizer.from_pretrained(token_type)
		nr, nc, dtype = open(f'{fname}.meta').readline().split()
		self.X_ii = np.memmap(f"{fname}", mode='r', shape=(int(nr), int(nc)), dtype=dtype)
	
	def get_fts(self, indices):
		X_ii = np.array(self.X_ii[self.sample[indices]])
		X_am = (X_ii != self.tokenizer.pad_token_id)
		return bert_fts_batch_to_tensor(X_ii, X_am)
	
class OnlineBertDataset(BaseDataset):
	def __init__(self, X, labels, max_len, token_type='bert-base-uncased', **super_kwargs):
		super().__init__(labels, **super_kwargs)
		self.max_len = max_len
		self.X = np.array(X, dtype=object)
		self.tokenizer = AutoTokenizer.from_pretrained(token_type)
	
	def get_fts(self, indices):
		return self.tokenizer.batch_encode_plus(list(self.X[self.sample[indices]]), 
												max_length=self.max_len, 
												padding=True, 
												truncation=True, 
												return_tensors='pt', 
												return_token_type_ids=False).data

class XMCCollator():
	def __init__(self, dataset):
		self.dataset = dataset
		self.numy = self.dataset.labels.shape[1]
	
	def __call__(self, batch):
		batch_size = len(batch)
		ids = torch.LongTensor([b['index'] for b in batch])
		
		b = {'batch_size': torch.LongTensor([batch_size]),
			 'numy': torch.LongTensor([self.numy]),
			 'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
			 'ids': ids,
			 'xfts': self.dataset.get_fts(ids)}

		return b

class TwoTowerDataset(BaseDataset):
	def __init__(self, x_dataset, y_dataset):
		super().__init__(labels=x_dataset.labels, filter_mat=x_dataset.filter_mat)
		self.x_dataset = x_dataset
		self.y_dataset = y_dataset
            
	def __getitem__(self, index):
		ret = {'index': index}
		return ret

	def get_fts(self, indices, source):
		if source == 'x':
			return self.x_dataset.get_fts(indices)
		elif source == 'y':
			return self.y_dataset.get_fts(indices)

	def __len__(self):
		return self.labels.shape[0]

class TwoTowerTrainCollator():
    def __init__(self, dataset: TwoTowerDataset, _type='batch-rand'):
        self.numy = dataset.labels.shape[1]
        self.dataset = dataset
        self._type = _type
        self.mask = torch.zeros(self.numy+1).long()
    
    def __call__(self, batch):
        batch_size = len(batch)
        ids = np.array([b['index'] for b in batch])
        
        batch_data = {'batch_size': batch_size,
                      'numy': self.numy,
                      'y': csr_to_pad_tensor(self.dataset.labels[ids], self.numy),
                      'ids': torch.Tensor([b['index'] for b in batch]).long(),
                      'xfts': self.dataset.get_fts(ids, 'x')
                     }
        
        batch_y = None
        
        if self._type == 'shorty':
            batch_data['shorty'] = csr_to_pad_tensor(self.dataset.shorty[ids], self.numy)
            batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), 1)
            batch_pos_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
            batch_y = torch.LongTensor(np.union1d(batch_pos_y, batch_data['shorty']['inds']))
            
            self.mask[batch_y] = torch.arange(batch_y.shape[0])
            batch_data['pos-inds'] = self.mask[batch_pos_y].reshape(-1, 1)
            batch_data['shorty']['batch-inds'] = self.mask[batch_data['shorty']['inds'].flatten()].reshape(batch_data['shorty']['inds'].shape)
            self.mask[batch_y] = 0
            
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
            for i in range(batch_size):
                self.mask[batch_data['y']['inds'][i]] = True
                batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
                self.mask[batch_data['y']['inds'][i]] = False
        
        elif self._type == 'batch-rand':
            batch_y_inds = torch.multinomial(batch_data['y']['vals'].double(), 1)
            batch_y = torch.gather(batch_data['y']['inds'], 1, batch_y_inds).squeeze()
            batch_data['pos-inds'] = torch.arange(batch_size).reshape(-1, 1)
            batch_data['targets'] = torch.zeros((batch_size, batch_y.shape[0]))
            for i in range(batch_size):
                self.mask[batch_data['y']['inds'][i]] = True
                batch_data['targets'][i][self.mask[batch_y].bool()] = 1.0
                self.mask[batch_data['y']['inds'][i]] = False
                
        if batch_y is not None:
            batch_data['batch_y'] = batch_y
            batch_data['yfts'] = self.dataset.get_fts(batch_y.numpy(), 'y')

        return batch_data

class XMCDataManager():
	def __init__(self, args):
		self.trn_X_Y = sp.load_npz(f'{args.DATA_DIR}/Y.trn.npz')
		self.tst_X_Y = sp.load_npz(f'{args.DATA_DIR}/Y.tst.npz')
		self.tst_filter_mat = load_filter_mat(f'{args.DATA_DIR}/filter_labels_test.txt', self.tst_X_Y.shape)
		self.trn_filter_mat = load_filter_mat(f'{args.DATA_DIR}/filter_labels_train.txt', self.trn_X_Y.shape)
		self.inv_prop = get_inv_prop(self.trn_X_Y, args.dataset)

		self.numy = args.numy = self.trn_X_Y.shape[1] # Number of labels
		self.trn_numx = self.trn_X_Y.shape[0] # Number of train data points 
		self.tst_numx = self.tst_X_Y.shape[0] # Number of test data points

		self.data_tokenization = args.data_tokenization
		self.tf_max_len = args.tf_max_len
		self.tf_token_type = args.tf_token_type = 'roberta-base' if 'roberta' in args.tf else 'bert-base-uncased' if 'bert' in args.tf else args.tf # Token type
		self.DATA_DIR = args.DATA_DIR
		self.num_val_points = args.num_val_points
		self.bsz = args.bsz

		if self.num_val_points > 0:
			if os.path.exists(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy'): 
				self.val_inds = np.load(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy')
			else: 
				self.val_inds = np.random.choice(np.arange(self.trn_numx), size=args.num_val_points, replace=False)
				np.save(f'{args.DATA_DIR}/val_inds_{args.num_val_points}.npy', self.val_inds)
			self.trn_inds = np.setdiff1d(np.arange(self.trn_numx), self.val_inds)
		else:
			self.trn_inds = self.val_inds = None

	def load_bow_fts(self, normalize=True):
		trn_X_Xf = sp.load_npz(f'{self.DATA_DIR}/X.trn.npz')
		tst_X_Xf = sp.load_npz(f'{self.DATA_DIR}/X.tst.npz')

		if normalize:
			sklearn.preprocessing.normalize(trn_X_Xf, copy=False)
			sklearn.preprocessing.normalize(tst_X_Xf, copy=False)

		self.trn_X_Xf = trn_X_Xf[self.trn_inds] if self.trn_inds is not None else trn_X_Xf
		self.val_X_Xf = trn_X_Xf[self.val_inds] if self.val_inds is not None else tst_X_Xf
		self.tst_X_Xf = tst_X_Xf

		return self.trn_X_Xf, self.val_X_Xf, self.tst_X_Xf

	def build_datasets(self):
		if self.data_tokenization == 'offline':
			self.trn_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/trn_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
			self.val_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/trn_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.val_inds, filter_mat=self.trn_filter_mat)
			self.tst_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/tst_X.{self.tf_token_type}_{self.tf_max_len}.dat', self.tst_X_Y, self.tf_max_len, self.tf_token_type, sample = None, filter_mat=self.tst_filter_mat)
		elif self.data_tokenization == 'online':
			trnX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/trn_X.txt').readlines()]
			tstX = [x.strip() for x in open(f'{self.DATA_DIR}/raw/tst_X.txt').readlines()]
			self.trn_dataset = OnlineBertDataset(trnX, self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.trn_inds, filter_mat=self.trn_filter_mat)
			self.val_dataset = OnlineBertDataset(trnX, self.trn_X_Y, self.tf_max_len, self.tf_token_type, sample=self.val_inds, filter_mat=self.trn_filter_mat)
			self.tst_dataset = OnlineBertDataset(tstX, self.tst_X_Y, self.tf_max_len, self.tf_token_type, sample=None, filter_mat=self.tst_filter_mat)
		else:
			raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
		
		if self.num_val_points <= 0:
			self.val_dataset = self.tst_dataset

		return self.trn_dataset, self.val_dataset, self.tst_dataset

	def build_data_loaders(self):
		if not hasattr(self, "trn_dataset"):
			self.build_datasets()

		data_loader_args = {
			'batch_size': self.bsz,
			'num_workers': 4,
			'collate_fn': XMCCollator(self.trn_dataset),
			'shuffle': True,
			'pin_memory': True
		}

		self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

		data_loader_args['shuffle'] = False
		data_loader_args['collate_fn'] = XMCCollator(self.val_dataset)
		data_loader_args['batch_size'] = 2*self.bsz
		self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)

		data_loader_args['collate_fn'] = XMCCollator(self.tst_dataset)
		self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

		return self.trn_loader, self.val_loader, self.tst_loader

class TwoTowerDataManager(XMCDataManager):
	def __init__(self, args):
		super().__init__(args)
		self.transpose_trn_dataset = args.transpose_trn_dataset

	def build_datasets(self):
		trnx_dataset, valx_dataset, tstx_dataset = super().build_datasets()
		if self.data_tokenization == 'offline':
			self.lbl_dataset = OfflineBertDataset(f'{self.DATA_DIR}/raw/Y.{self.tf_token_type}_{self.tf_max_len}.dat', self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=None, filter_mat=None)
		elif self.data_tokenization == 'online':
			Y = [x.strip() for x in open(f'{self.DATA_DIR}/raw/Y.txt').readlines()]
			self.lbl_dataset = OnlineBertDataset(Y, self.trn_X_Y.T.tocsr(), self.tf_max_len, self.tf_token_type, sample=None, filter_mat=None)
		else:
			raise Exception(f"Unrecongnized data_tokenization argument: {self.data_tokenization}")
		
		if self.transpose_trn_dataset:
			self.trn_dataset = TwoTowerDataset(self.lbl_dataset, trnx_dataset)
		else:
			self.trn_dataset = TwoTowerDataset(trnx_dataset, self.lbl_dataset)
		self.val_dataset = TwoTowerDataset(valx_dataset, self.lbl_dataset)
		self.tst_dataset = TwoTowerDataset(tstx_dataset, self.lbl_dataset)

		return self.trn_dataset, self.val_dataset, self.tst_dataset

	def build_data_loaders(self):
		if not hasattr(self, "trn_dataset"):
			self.build_datasets()

		data_loader_args = {
			'batch_size': self.bsz,
			'num_workers': 4,
			'collate_fn': TwoTowerTrainCollator(self.trn_dataset),
			'shuffle': True,
			'pin_memory': True
		}

		self.trn_loader = torch.utils.data.DataLoader(self.trn_dataset, **data_loader_args)

		data_loader_args['shuffle'] = False
		data_loader_args['collate_fn'] = None
		data_loader_args['batch_size'] = 2*self.bsz
		self.val_loader = torch.utils.data.DataLoader(self.val_dataset, **data_loader_args)
		self.tst_loader = torch.utils.data.DataLoader(self.tst_dataset, **data_loader_args)

		return self.trn_loader, self.val_loader, self.tst_loader

class XMCEvaluator:
	def __init__(self, args, data_source, data_manager: XMCDataManager, prefix='default'):
		self.eval_interval = args.eval_interval
		self.num_epochs = args.num_epochs
		self.track_metric = args.track_metric
		self.OUT_DIR = args.OUT_DIR
		self.save = args.save
		self.bsz = args.bsz
		self.eval_topk = args.eval_topk
		self.wandb_id = args.wandb_id if hasattr(args, "wandb_id") else None
		self.prefix = prefix

		self.data_source = data_source
		self.labels = data_source.labels if isinstance(data_source, torch.utils.data.Dataset) else data_source.dataset.labels
		self.filter_mat = data_source.filter_mat if isinstance(data_source, torch.utils.data.Dataset) else data_source.dataset.filter_mat
		self.inv_prop = data_manager.inv_prop
		self.best_score = -99999999

	def predict(self, net):
		score_mat = unwrap(net).predict(self.data_source, K=self.eval_topk, bsz=self.bsz)
		return score_mat

	def eval(self, score_mat, epoch=-1, loss=float('inf')):
		_filter(score_mat, self.filter_mat, copy=False)
		eval_name = f'{self.prefix}' + [f' {epoch}/{self.num_epochs}', ''][epoch < 0]
		metrics = compute_xmc_metrics(score_mat, self.labels, self.inv_prop, K=self.eval_topk, name=eval_name, disp=False)
		metrics.index.names = [self.wandb_id]
		if loss < float('inf'):  metrics['loss'] = ["%.4E"%loss]
		metrics.to_csv(open(f'{self.OUT_DIR}/{self.prefix}_metrics.tsv', 'a+'), sep='\t', header=(epoch <= 0))
		return metrics

	def predict_and_track_eval(self, net, epoch='-', loss=float('inf')):
		if epoch%self.eval_interval == 0 or epoch == (self.num_epochs-1):
			score_mat = self.predict(net)
			metrics = self.eval(score_mat, epoch, loss)
			if metrics.iloc[0][self.track_metric] > self.best_score:
				self.best_score = metrics.iloc[0][self.track_metric]
				print(_c(f'Found new best model with {self.track_metric}: {"%.2f"%self.best_score}\n', attr='blue'))
				if self.save:
					sp.save_npz(f'{self.OUT_DIR}/{self.prefix}_score_mat.npz', score_mat)
					net.save(f'{self.OUT_DIR}/model.pt')
			return metrics

DATA_MANAGERS = {
	'xmc': XMCDataManager,
	'two-tower': TwoTowerDataManager
}
			