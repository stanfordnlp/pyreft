
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer

if __name__ == '__main__':


    print('Downloading checkpoints if not cached')
    print('T5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base', cache_dir="/nlp/scr/peterwz/.cache")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    print('BART-base')
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", cache_dir="/nlp/scr/peterwz/.cache")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    print('Done!')

