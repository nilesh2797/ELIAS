from accelerate import Accelerator
accelerator = Accelerator()
IS_MAIN_PROC = accelerator.is_main_process

import sys, yaml, logging
import logging.config
import scipy.sparse as sp

from nets import *
from resources import _c, load_config_and_runtime_args
from datasets import DATA_MANAGERS, XMCEvaluator
from dl_helper import unwrap
if torch.__version__ > "1.11":
    torch.backends.cuda.matmul.allow_tf32 = True

import transformers
transformers.set_seed(42)

# Config and runtime argument parsing
args = load_config_and_runtime_args(sys.argv)
    
args.device = str(accelerator.device)
args.amp = accelerator.state.use_fp16
args.num_gpu = accelerator.state.num_processes
args.DATA_DIR = DATA_DIR = f'Datasets/{args.dataset}'
args.resume_path = f'{args.OUT_DIR}/model.pt'

if IS_MAIN_PROC:
    with open('configs/logging.yaml') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = f"{args.OUT_DIR}/{log_config['handlers']['file_handler']['filename']}"
        logging.config.dictConfig(log_config)

logging.info(f'Starting {" ".join(sys.argv)}')
logging.info(f'Experiment name: {_c(args.expname, attr="blue")}, Dataset: {_c(args.dataset, attr="blue")}')
logging.info(f'Wandb ID: {args.wandb_id}')

# Data loading
data_manager = DATA_MANAGERS[args.data_manager](args)
_, _, tst_loader = data_manager.build_data_loaders()

accelerator.wait_for_everyone()
net = NETS[args.net](args)
logging.info(f'Loading net state dict from: {args.resume_path}')
logging.info(net.load(args.resume_path))
net, tst_loader = accelerator.prepare(net, tst_loader)

K = args.eval_topk
tst_X_Y = data_manager.tst_X_Y

net.eval()
score_mat = unwrap(net).predict(tst_loader, bsz=args.bsz*2, K=args.eval_topk, accelerator=accelerator)

if IS_MAIN_PROC: 
    evaluator = XMCEvaluator(args, tst_loader, data_manager, prefix='tst')
    metrics = evaluator.eval(score_mat)
    logging.info('\n'+metrics.to_csv(sep='\t', index=False))
    sp.save_npz(f'{args.OUT_DIR}/{evaluator.prefix}_score_mat.npz', score_mat)

accelerator.wait_for_everyone()