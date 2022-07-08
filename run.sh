seed=0
dataset=amazon-670k
expname=Stage-1-${seed}-test

# accelerate launch --config_file accl_config.yaml aux.py --dataset amazon-670k --expname ${expname} --tf bert-base-uncased --maxlen 128 --gen-cmat

# python train.py --expname ${expname} --dataset amazon-670k --maxlen 128 --net elias --stage 1 --tf bert-base-uncased --loss joint --xc-lr 2e-2 --enc-lr 2e-4 --bsz 512 --W-accum-steps 10 --save --num-epochs 5 --cmat-seed ${seed} --joint-loss-gamma 0 --no-cmat-update --A-init-file Results/ELIAS/${dataset}/${expname}/cmat.npz

# accelerate launch --config_file accl_config.yaml aux.py --dataset amazon-670k --expname ${expname} --gen-cmat

# python train.py --expname ${expname} --dataset amazon-670k --maxlen 128 --net elias --stage 1 --tf bert-base-uncased --loss joint --xc-lr 2e-2 --enc-lr 2e-4 --bsz 512 --W-accum-steps 10 --save --num-epochs 15 --cmat-seed ${seed} --joint-loss-gamma 0 --no-cmat-update --A-init-file Results/ELIAS/${dataset}/${expname}/cmat.npz

# accelerate launch --config_file accl_config.yaml aux.py --dataset amazon-670k --expname ${expname} --gen-A

expname=Stage-2-${seed}-test
python train.py --expname ${expname} --dataset ${dataset} --maxlen 128 --net elias --stage 2 --tf bert-base-uncased --loss joint --xc-lr 1e-2 --dense-lr 5e-4 --enc-lr 5e-5 --bsz 512 --W-accum-steps 10 --resume Results/ELIAS/${dataset}/Stage-1-${seed}-test/model.pt --num-epochs 40 --use-swa --swa-step 2000 --swa-start 6 --cmat-seed ${seed} --save --save-embs --ranker

# accelerate launch --config_file accl_config.yaml engine.py --expname Stage-2-${seed}-parallel --dataset ${dataset} --maxlen 128 --net elias --stage 2 --tf bert-base-uncased --loss joint --joint-loss-gamma 0.05 --xc-lr 3e-3 --dense-lr 1e-4 --enc-lr 2e-5 --bsz 512 --W-accum-steps 1 --resume Results/ELIAS/${dataset}/Stage-1-${seed}/model.pt --num-epochs 40 --cmat-seed ${seed} --save --no-amp-encode

# accelerate launch --config_file accl_config.yaml gen_A_init.py --dataset ${dataset} --exp Stage-1-0 --num-epochs 0

# python engine.py --expname Stage-1-${seed} --dataset ${dataset} --maxlen 32 --net elias --stage 1 --tf distilbert-base-uncased --loss joint --xc-lr 1e-1 --enc-lr 1e-4 --bsz 1024 --W-accum-steps 10 --save --save-embs --gen-A-init --num-epochs 20 --cmat-seed ${seed} --joint-loss-gamma 0

# python engine.py --expname Stage-2-${seed}-no-swa --dataset ${dataset} --maxlen 32 --net elias --stage 2 --tf distilbert-base-uncased --loss joint --xc-lr 5e-2 --dense-lr 5e-4 --enc-lr 5e-5 --bsz 1024 --W-accum-steps 10 --resume Results/ELIAS/${dataset}/Stage-1-${seed}/model.pt --num-epochs 40 --cmat-seed ${seed} --save --save-embs --ranker

# accelerate launch --config_file accl_config.yaml engine.py --expname Stage-1-${seed} --dataset ${dataset} --maxlen 32 --net elias --stage 1 --tf distilbert-base-uncased --loss joint --xc-lr 0.03 --enc-lr 3e-4 --bsz 1024 --W-accum-steps 1 --num-epochs 20 --cmat-seed ${seed} --joint-loss-gamma 0.01 --no-cmat-update --no-amp-encode --eval-interval 1 --A-init-file Datasets/${dataset}/cmat-1024x100-0.npz --clf-dim 64 --save

# accelerate launch --config_file accl_config.yaml engine.py --expname Stage-1-${seed}-qbst --dataset ${dataset} --maxlen 64 --net elias --stage 1 --tf distilbert-base-uncased --loss joint --xc-lr 0.03 --enc-lr 3e-4 --bsz 512 --W-accum-steps 1 --num-epochs 20 --cmat-seed ${seed} --joint-loss-gamma 0.01 --no-cmat-update --no-amp-encode --eval-interval 1 --A-init-file Datasets/${dataset}/cmat-131072x100-0.npz --clf-dim 64 --save

# accelerate launch --config_file accl_config.yaml engine.py --expname Stage-2-${seed}-qbst-no-swa --dataset ${dataset} --maxlen 64 --net elias --stage 2 --tf distilbert-base-uncased --loss joint --joint-loss-gamma 0.03 --xc-lr 1e-2 --dense-lr 5e-5 --enc-lr 1e-5 --bsz 320 --W-accum-steps 1 --resume Results/ELIAS/${dataset}/Stage-1-${seed}-qbst/model.pt --num-epochs 25 --cmat-seed ${seed} --save --clf-dim 64 --eval-interval 1 --no-amp-encode

# accelerate launch --config_file accl_config.yaml engine.py --expname Stage-2-${seed}-no-swa --dataset ${dataset} --maxlen 32 --net elias --stage 2 --tf distilbert-base-uncased --loss joint --xc-lr 2e-2 --dense-lr 1e-4 --enc-lr 2e-5 --bsz 512 --W-accum-steps 1 --resume Results/ELIAS/${dataset}/Stage-2-${seed}-no-swa/model.pt --num-epochs 25 --cmat-seed ${seed} --save --clf-dim 64 --eval-interval 1 --no-amp-encode --A-init-file Results/ELIAS/${dataset}/Stage-1-${seed}/A_init.npz

# python engine.py --expname temp --dataset ${dataset} --maxlen 32 --net elias --stage 2 --tf distilbert-base-uncased --loss joint --xc-lr 2e-2 --dense-lr 1e-4 --enc-lr 5e-5 --bsz 256 --W-accum-steps 10 --resume Results/ELIAS/${dataset}/Stage-1-${seed}/model.pt --num-epochs 25 --cmat-seed ${seed} --clf-dim 64 --eval-interval 1 --no-amp-encode

# accelerate launch --config_file accl_config.yaml gen_A_init.py --dataset ShoppingAds-10M-v2 --expname Stage-1-0-qbst --num-epochs 0

#----------------------------------------------------------------------------------------------------------------------#

# seed=0
# dataset=amazon-670k

# accelerate launch --config_file accl_config.yaml engine.py --expname Stage-1-${seed} --dataset ${dataset} --maxlen 128 --net elias --stage 1 --tf distilbert-base-uncased --loss joint --xc-lr 0.01 --enc-lr 2e-4 --bsz 256 --W-accum-steps 1 --num-epochs 20 --cmat-seed ${seed} --joint-loss-gamma 0 --no-cmat-update --no-amp-encode --A-init-file Datasets/amazon-670k/cmat-8192x100-0.npz

# python engine.py --expname Stage-1-${seed} --dataset ${dataset} --maxlen 128 --net elias --stage 1 --tf distilbert-base-uncased --loss joint --xc-lr 0.005 --enc-lr 1e-4 --bsz 256 --W-accum-steps 1 --num-epochs 20 --cmat-seed ${seed} --joint-loss-gamma 0 --no-cmat-update --A-init-file Datasets/amazon-670k/cmat-8192x100-0.npz