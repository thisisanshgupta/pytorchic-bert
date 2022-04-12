
# Pytorch BERT
## Requirements

Python > 3.6, fire, tqdm, tensorboardx,
tensorflow (for loading checkpoint file)

## Data

1. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) caption data. Extract them to input for pretraining birds corpus data.

## Example Usage


### Pre-training Transformer
Input file format :
1. One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text. (Because we use the sentence boundaries for the "next sentence prediction" task).
2. Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task doesn't span between documents.
```
a large bird has curly hair on its nape, and a long and sharp bill that is black.
a bird with a long bill that curves downwards and small eyes
this is a brown bird with a black face and a large pointy black beak.
this large bird has brown fur and a large black beak
this is a bird that appears to be young with brown patchy downy feathers and a long hooked bill and black face.
a dark colored bird with short fuzzy feathers starting toward the back of its crown and a bald face and throat and a long black bill that is open and the top curves down at the end.
this particular bird has a belly that is brown and black
this bird has a long black bill and has brown fur
this bird has a very long bill that tapers down to a point, it has black eyes and brown fluff on it's crown.
this bird has wings that are brown and has a long black bill

a bird with multiple colors, brown, white and black with an orange spot above its eye
this is a white and yellow bird with black spots and a pointy beak.
a small brown and light orange bird with small black eyes.
this bird is black with white on its chest and has a long, pointy beak.
this bird is brown and black in color with a grey beak, and light eye rings.
this slim bird has a small bill, with the same curvature as its head, and a long tail.
this little bird's colors allows it to easily camouflage with the brush.
the bird has a spotted wingbar that is brown and a small bill.
the bird has a black eyering and a spotted brown wingbar.
this bird has a brown crown, brown primaries, and a tan belly.

a small bird with a white underbelly and black and white striped crown.
this is a small brown bird with white eyebroews, and a very small beak.
...
```

making BERT vocab :
```
import sentencepiece as spm

parameter = '--input={} --model_prefix={} --vocab_size={} --user_defined_symbols={}'

input_file = 'birds_corpus.txt'
vocab_size = 3765
prefix = 'sentpiece_vocab'
user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK]'
cmd = parameter.format(input_file, prefix, vocab_size,user_defined_symbols)

spm.SentencePieceTrainer.Train(cmd)
```

Usage :
```
!python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file ./data/birds_corpus.txt \
    --vocab ./pretrain/vocab.txt \
    --save_dir ./output \
    --log_dir ./output/runs \
    --max_len 64 \
    --max_pred 20 \
    --mask_prob 0.15
