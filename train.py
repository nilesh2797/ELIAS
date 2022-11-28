# Imports
from accelerate import Accelerator
accelerator = Accelerator()
IS_MAIN_PROC = accelerator.is_main_process

import sys, os, time, socket, yaml, wandb, logging
import logging.config
from tqdm import tqdm

from nets import *
from losses import *
from optimizer_bundles import *
from resources import _c, load_config_and_runtime_args, dump_diff_config
from datasets import DATA_MANAGERS, XMCEvaluator
from dl_helper import unwrap

import torch
import transformers
transformers.set_seed(42)
if torch.__version__ > "1.11":
    torch.backends.cuda.matmul.allow_tf32 = True

# Config and runtime argument parsing
args = load_config_and_runtime_args(sys.argv)
    
args.device = str(accelerator.device)
args.amp = accelerator.state.use_fp16
args.num_gpu = accelerator.state.num_processes
args.use_grad_scaler = ((not args.amp) and (args.num_gpu == 1) and args.amp_encode)
args.hostname = socket.gethostname()
args.exp_start_time = time.ctime()
args.embs_dim = 768 # TODO(@nilesh): remove this hard coded value
args.DATA_DIR = DATA_DIR = f'Datasets/{args.dataset}'
args.OUT_DIR = OUT_DIR = f'Results/{args.project}/{args.dataset}/{args.expname}'
os.makedirs(OUT_DIR, exist_ok=True)

if IS_MAIN_PROC:
    with open('configs/logging.yaml') as f:
        log_config = yaml.safe_load(f.read())
        log_config['handlers']['file_handler']['filename'] = f"{OUT_DIR}/{log_config['handlers']['file_handler']['filename']}"
        logging.config.dictConfig(log_config)

logging.info(f'Starting run {" ".join(sys.argv)}')
logging.info(f'Experiment name: {_c(args.expname, attr="blue")}, Dataset: {_c(args.dataset, attr="blue")}')

# Data loading
data_manager = DATA_MANAGERS[args.data_manager](args)
trn_loader, val_loader, _ = data_manager.build_data_loaders()

if IS_MAIN_PROC:
    wandb.init(project=f'{args.project}_{args.dataset}', name=args.expname)
    wandb.config.update(args)
    args.wandb_id = wandb.run.id
    logging.info(f'Wandb ID: {args.wandb_id}')
    logging.info(dump_diff_config(f'{OUT_DIR}/config.yaml', args.__dict__))
    evaluator = XMCEvaluator(args, val_loader, data_manager, 'val')

# Preparing network, loss, and optimizer object
accelerator.wait_for_everyone()
net = NETS[args.net](args)
criterion = LOSSES[args.loss](args)
optim_bundle = OPTIM_BUNDLES[args.optim_bundle](args)

if os.path.exists(args.resume_path):
    logging.info(f'Loading net state dict from: {args.resume_path}')
    logging.info(net.load(args.resume_path))

optim_bundle.inject_params(net)
net, optim_bundle.optims, trn_loader = accelerator.prepare(net, optim_bundle.optims, trn_loader)
optim_bundle.init_schedulers(args, len(trn_loader))
logging.info(optim_bundle)

# Training loop
scaler = torch.cuda.amp.GradScaler()
global_step = 0

for epoch in range(args.num_epochs):
    epoch_loss = 0
    net.train()
    optim_bundle.zero_grad()

    t = tqdm(trn_loader, desc='Epoch: 0, Loss: 0.0', leave=True, disable=not IS_MAIN_PROC)
    for i, b in enumerate(t):
        loss = criterion(net, b)
        scaler.scale(loss.float()).backward() if args.use_grad_scaler else accelerator.backward(loss)
        
        optim_bundle.step_and_zero_grad(scaler if args.use_grad_scaler else None)
        unwrap(net).update_non_parameters(epoch, global_step)

        epoch_loss += loss.item()
        global_step += 1
        t.set_description('Epoch: %d/%d, Loss: %.4E'%(epoch, args.num_epochs, (epoch_loss/(i+1))), refresh=True)
        
    if IS_MAIN_PROC:
        epoch_loss = (epoch_loss/(i+1))
        wandb.log({'epoch': epoch, 'loss': epoch_loss}, step=epoch)
        logging.info(f'Mean loss after epoch {epoch}/{args.num_epochs}: {"%.4E"%(epoch_loss)}')
        metrics = evaluator.predict_and_track_eval(unwrap(net), epoch, epoch_loss)
        if metrics is not None:
            logging.info('\n'+metrics.to_csv(sep='\t', index=False))
            wandb.log({k: metrics.iloc[0][k] for k in ['P@1', 'P@5', 'PSP@1', 'PSP@5', 'nDCG@5', 'R@100']}, step=epoch)
            wandb.log({'metrics': metrics}, step=epoch)
        sys.stdout.flush()
    accelerator.wait_for_everyone()
    torch.cuda.empty_cache()
