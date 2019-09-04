import json
import regex as re
import pandas as pd
import urllib.request
import os
from functools import lru_cache
from azureml.studio.score import ioutil

from pip._internal import main as pipmain
pipmain(["install", "click"])
import click

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logging.info(f"Encode text with BPE")
logger = logging.getLogger(__name__)

INPUT_FILE_NAME = "data.dataset.parquet" # hard coded, to be replaced, and we presume the data is DataFrame inside parquet
DICT_PATH_KEY = "Dictionary Path"
VOCAB_PATH_KEY = "Vocabulary Path"

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPEEncoder(object):
    def __init__(self, params={}):
        print(f"BPEEncoder({params})")
        dict_path = params.get(
            DICT_PATH_KEY, None
        )
        vocab_path = params.get(
            VOCAB_PATH_KEY, None
        )
        self.byte_encoder = bytes_to_unicode()
        response = urllib.request.urlopen(dict_path)
        self.dict = json.load(response)
        response = urllib.request.urlopen(vocab_path)
        bpe_vocab = response.read().decode('utf-8')
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_vocab.split('\n')[1:-1]]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.decoder = {v:k for k,v in self.dict.items()}
        self.errors = 'replace'  # how to handle errors in decoding

    def encode(self, input_df):
        print(f"BPE encoder input({input_df})")
        raw_text = ''.join(input_df.values.flatten())
        bpe_tokens = []
        for token in re.findall(self.pat, raw_text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.dict[bpe_token] for bpe_token in self.bpe(token).split(' '))
        print(f'result: {bpe_tokens}')
        df = pd.DataFrame()
        df["input:0"] = [bpe_tokens]
        return df

    def bpe(self, token):
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        return word

    def decode(self, input_df):
        # Change shape for built-in score
        context_tokens = input_df.values.flatten()[0]
        text = ''.join([self.decoder[token] for token in context_tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        print(f'RESULT: {text}')
        output = pd.DataFrame([text])
        return output


@click.command()
@click.option('--dict_path')
@click.option('--vocab_path')
@click.option('--input_text_path', default="inputs/gpt2")
@click.option('--output_path', default="outputs/gpt2/encoded_text")
def run_pipeline(dict_path, vocab_path, input_text_path, output_path):
    input_df = pd.read_parquet(os.path.join(input_text_path, INPUT_FILE_NAME), engine="pyarrow")
    meta = {
        DICT_PATH_KEY: dict_path,
        VOCAB_PATH_KEY: vocab_path
    }
    encoder = BPEEncoder(meta)
    df = encoder.encode(input_df)
    ioutil.save_parquet(df, output_path)
    print(f'Output path: {os.listdir(output_path)}')


# TODO(wanhan): use a better way to store dict and vocabulary
# python -m examples.tensorflow.gpt2.encode ...
if __name__ == '__main__':
    run_pipeline()
