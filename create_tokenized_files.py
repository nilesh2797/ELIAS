"""
For a dataset, create tokenized files in the folder {tokenizer-type}-{maxlen} folder inside the database folder
Sample usage: python -W ignore -u CreateTokenizedFiles.py --data-dir Datasets/LF-AmazonTitles-131K --max-length 32
"""
import os, time
import torch.multiprocessing as mp
from transformers import AutoTokenizer
import numpy as np
import functools
import argparse
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
def timeit(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      
        run_time = end_time - start_time
        print("Finished {} in {:.4f} secs".format(func.__name__, run_time))
        return value
    return wrapper_timer

def _tokenize(batch_input):
    tokenizer, maxlen, batch_corpus = batch_input[0], batch_input[1], batch_input[2]
    temp = tokenizer.batch_encode_plus(
                    batch_corpus,                           # Sentence to encode.
                    add_special_tokens = True,              # Add '[CLS]' and '[SEP]'
                    max_length = maxlen,                   # Pad & truncate all sentences.
                    padding = 'max_length',
                    return_attention_mask = True,           # Construct attn. masks.
                    return_tensors = 'np',                  # Return numpy tensors.
                    truncation=True
            )

    return (temp['input_ids'], temp['attention_mask'])

def convert(corpus, tokenizer, maxlen, num_threads):
    bsz = len(corpus)//num_threads 
    batches = [(tokenizer, maxlen, corpus[batch_start: batch_start + bsz]) for batch_start in range(0, len(corpus), bsz)]

    pool = mp.Pool(num_threads)
    batch_tokenized = pool.map(_tokenize, batches)
    pool.close()

    input_ids = np.vstack([x[0] for x in batch_tokenized])
    attention_mask = np.vstack([x[1] for x in batch_tokenized])

    del batch_tokenized 

    return input_ids, attention_mask

@timeit
def tokenize_dump_memmap(corpus, output_path, tokenizer, maxlen, num_threads, batch_size=500000):
    ii = np.memmap(output_path, dtype='int64', mode='w+', shape=(len(corpus), maxlen))
    for i in tqdm(range(0, len(corpus), batch_size)):
        _input_ids, _ = convert(corpus[i: i + batch_size], tokenizer, maxlen, num_threads)
        ii[i: i + _input_ids.shape[0], :] = _input_ids
    print(len(corpus), maxlen, 'int64', file=open(f'{output_path}.meta', 'w'))

parser = argparse.ArgumentParser()

parser.add_argument("--data-path", type=str, required=True, help="Data path")
parser.add_argument("--tf-max-len", type=int, help="Max length for tokenizer", default=32)
parser.add_argument("--tf-token-type", type=str, help="Tokenizer to use", default="bert-base-uncased")
parser.add_argument("--num-threads", type=int, help="Number of threads to use", default=32)

args = parser.parse_args()

lines = [x.strip() for x in open(f'{args.data_path}', "r", encoding="utf-8").readlines()]
print(f'Read {len(lines)} lines')
maxlen = args.tf_max_len

tokenizer = AutoTokenizer.from_pretrained(args.tf_token_type, do_lower_case=True)

output_path = f'{".".join(args.data_path.split(".")[:-1])}.{args.tf_token_type}_{args.tf_max_len}.dat'
print(f"Dumping tokenized file at {output_path}...")

tokenize_dump_memmap(lines, output_path, tokenizer, maxlen, args.num_threads)
