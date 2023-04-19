dataset=$1
num_gpus=$2
project="ELIAS"
# Generate initial clustering with BOW features
python elias_utils.py gen_cluster_A configs/${dataset}/elias-1-def-final.yaml --no_model true
sleep 100
# Train ELIAS-1 model for a small number (5) of epochs
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/elias-1-def-final.yaml --num_epochs 5
sleep 100
# Generate clusters again, this time with concatenated BOW features and dense embedding from partially trained ELIAS-1 model
python elias_utils.py gen_cluster_A Results/${project}/${dataset}/elias-1-def-final/config.yaml
# Train ELIAS-1 model with fixed initial clusters
sleep 100
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/elias-1-def-final.yaml
# ELIAS-1 results
echo "Elias 1 results"
sleep 100
CUDA_VISIBLE_DEVICES=0 python eval.py Results/${project}/${dataset}/elias-1-def-final/config.yaml
# Generate row-wise sparse approximate adjacency graph (A) between clusters and labels based on trained ELIAS-1 model
sleep 100
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} elias_utils.py gen_approx_A Results/${project}/${dataset}/elias-1-def-final/config.yaml
# Train ELIAS-2 model (it takes trained ELIAS-1 model and generated approximate A as initialization and trains them end-to-end)
sleep 100
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} train.py configs/${dataset}/elias-2-def-final.yaml
# ELIAS-2 Results
sleep 100
echo "ELIAS 2 Results"
python eval.py Results/${project}/${dataset}/elias-2-def-final/config.yaml
# Evaluate trained model
sleep 100
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} eval.py Results/${project}/${dataset}/elias-2-def-final/config.yaml
#echo "Training sparse ranker"
#python elias_utils.py sparse_ranker Results/${project}/${dataset}/elias-2-def-final/config.yaml
