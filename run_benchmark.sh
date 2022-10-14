dataset=$1

# Generate initial clustering with BOW features
python elias_utils.py gen_cluster_A configs/${dataset}/elias-1.yaml --no_model true
# Train ELIAS-1 model for a small number (5) of epochs
python train.py configs/${dataset}/elias-1.yaml --num_epochs 5
# Generate clusters again, this time with concatenated BOW features and dense embedding from partially trained ELIAS-1 model
python elias_utils.py gen_cluster_A Results/ELIAS/${dataset}/ELIAS-1/config.yaml
# Train ELIAS-1 model with new clusters
python train.py configs/${dataset}/elias-1.yaml
# Generate row-wise sparse approximate adjacency graph (A) between clusters and labels based on trained ELIAS-1 model
python elias_utils.py gen_approx_A Results/ELIAS/${dataset}/ELIAS-1/config.yaml
# Train ELIAS-2 model (it takes trained ELIAS-1 model and generated approximate A as initialization and trains them end-to-end)
accelerate launch --config_file configs/accelerate.yaml train.py configs/${dataset}/elias-2.yaml
# Learn sparse ranker to re-rank top 100 predictions (only intended to get best precision numbers for paper, not very useful for retrieval or production)
python elias_utils.py sparse_ranker Results/ELIAS/${dataset}/ELIAS-2/config.yaml
