dataset=$1
num_gpus=4
project="ELIAS"
# Generate initial clustering with BOW features
python elias_utils.py gen_cluster_A configs/${dataset}/elias-1.yaml --no_model true
# Train ELIAS-1 model with fixed initial clusters
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/elias-1.yaml
# Generate row-wise sparse approximate adjacency graph (A) between clusters and labels based on trained ELIAS-1 model
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} elias_utils.py gen_approx_A Results/${project}/${dataset}/ELIAS-1/config.yaml
# Train ELIAS-2 model (it takes trained ELIAS-1 model and generated approximate A as initialization and trains them end-to-end)
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/elias-2.yaml
# Evaluate trained model
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} eval.py Results/${project}/${dataset}/ELIAS-2/config.yaml
