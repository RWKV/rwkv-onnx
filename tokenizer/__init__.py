from .tokenizer import RWKV_TOKENIZER, neox

fname = "rwkv_vocab_v20230424.txt"
world = RWKV_TOKENIZER(__file__[:__file__.rindex('/')] + '/' + fname)
neox = neox