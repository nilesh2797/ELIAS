<img src="media/neurips_logo.png" height="50" align="right"/>

# ELIAS
Learnable graph-based search index for large output spaces
<p align="center"><img src="media/elias_model.jpg" height="200"/></p>

> [End-to-end Learning to Index and Search in Large Output Spaces](https://arxiv.org/pdf/2210.08410.pdf) <br>
> Nilesh Gupta, Patrick H. Chen, Hsiang-Fu Yu, Cho-Jui Hsieh, Inderjit S. Dhillon <br>
> Neurips 2022
## Preparing data

## Download pretrained model 
Coming soon...
## Predicting ELIAS
```shell
# Single GPU
python eval.py ${config_dir}/config.yaml

# Multi GPU
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} eval.py ${config_dir}/config.yaml
```
## Training ELIAS
Sample script: [run_benchmark.sh](./run_benchmark.sh) (example `./run_benchmark.sh amazon-670k`)
### *Generate initial clustering matrix*
```shell
python elias_utils.py gen_cluster_A configs/${dataset}/elias-1.yaml --no_model true
```
### *Train Stage 1*
```shell
# Single GPU
python train.py configs/${dataset}/elias-1.yaml
# Multi GPU
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} eval.py configs/${dataset}/elias-1.yaml
```
### *Generate sparse approx adjacency graph matrix*
```shell
# Single GPU
python elias_utils.py gen_approx_A configs/${dataset}/elias-1.yaml
# Multi GPU
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} elias_utils.py gen_approx_A configs/${dataset}/elias-1.yaml
```
### *Train Stage 2*
```shell
# Single GPU
python train.py configs/${dataset}/elias-2.yaml
# Multi GPU
accelerate launch --config_file configs/accelerate.yaml --num_processes ${num_gpus} eval.py configs/${dataset}/elias-2.yaml
```
## Cite
```bib
@InProceedings{ELIAS,
  author    = "Gupta, N. and Chen, P.H. and Yu, H-F. and Hsieh, C-J. and Dhillon, I.",
  title     = "End-to-end Learning to Index and Search in Large Output Spaces",
  booktitle = "Neural Information Processing Systems",
  month     = "November",
  year      = "2022"
}
```