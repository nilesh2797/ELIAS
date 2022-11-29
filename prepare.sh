dataset=$1
max_seq_len=$2
token_type=$3

python create_tokenized_files.py --data-path Datasets/${dataset}/raw/trn_X.txt --tf-max-len ${max_seq_len} --tf-token-type ${token_type}
python create_tokenized_files.py --data-path Datasets/${dataset}/raw/tst_X.txt --tf-max-len ${max_seq_len} --tf-token-type ${token_type}