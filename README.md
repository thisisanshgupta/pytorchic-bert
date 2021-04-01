
# Pytorchic BERT revision and demos
This is Pytorchic BERT revised version for Google Colab.

Refer to [https://github.com/dhlee347/pytorchic-bert](https://github.com/dhlee347/pytorchic-bert) for original source version,<br>
it is implementation for the paper [Google BERT model](https://arxiv.org/abs/1810.04805).

Refer to troubleshooting [issues](https://github.com/rightlit/pytorchic-bert-rev/issues) while running with original source 

## Requirements

Python > 3.6, fire, tqdm, tensorboardx,
tensorflow (for loading checkpoint file)


## Example Usage


### Pre-training Transformer
Input file format :
1. One sentence per line. These should ideally be actual sentences, not entire paragraphs or arbitrary spans of text. (Because we use the sentence boundaries for the "next sentence prediction" task).
2. Blank lines between documents. Document boundaries are needed so that the "next sentence prediction" task doesn't span between documents.
```
Document 1 sentence 1
Document 1 sentence 2
...
Document 1 sentence 45

Document 2 sentence 1
Document 2 sentence 2
...
Document 2 sentence 24
```
Usage :
```
export DATA_FILE=/path/to/corpus
export BERT_PRETRAIN=/path/to/pretrain
export SAVE_DIR=/path/to/save

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --max_len 512 \
    --max_pred 20 \
    --mask_prob 0.15


