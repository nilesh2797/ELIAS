dataset=$1
max_seq_len=$2
token_type=$3

# mv Datasets/xmc-base/${dataset} Datasets/
# mv Datasets/${dataset}/tfidf-attnxml/* Datasets/${dataset}/
# mkdir -p Datasets/${dataset}/raw
# mv Datasets/${dataset}/X.trn.txt Datasets/${dataset}/raw/trn_X.txt
# mv Datasets/${dataset}/X.tst.txt Datasets/${dataset}/raw/tst_X.txt
# mv Datasets/${dataset}/output-items.txt Datasets/${dataset}/raw/Y.txt

python create_tokenized_files.py --data-path Datasets/${dataset}/raw/trn_X.txt --tf-max-len ${max_seq_len} --tf-token-type ${token_type}
python create_tokenized_files.py --data-path Datasets/${dataset}/raw/tst_X.txt --tf-max-len ${max_seq_len} --tf-token-type ${token_type}
python create_tokenized_files.py --data-path Datasets/${dataset}/raw/Y.txt --tf-max-len ${max_seq_len} --tf-token-type ${token_type}
