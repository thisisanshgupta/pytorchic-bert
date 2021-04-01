
# Pytorchic BERT
This is re-implementation of [Google BERT model](https://github.com/google-research/bert) [[paper](https://arxiv.org/abs/1810.04805)] in Pytorch. I was strongly inspired by [Hugging Face's code](https://github.com/huggingface/pytorch-pretrained-BERT) and I referred a lot to their codes, but I tried to make my codes **more pythonic and pytorchic style**. Actually, the number of lines is less than a half of HF's. 

(It is still not so heavily tested - let me know when you find some bugs.)

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
```
Output (with Toronto Book Corpus):
```
cuda (8 GPUs)
Iter (loss=5.837): : 30089it [18:09:54,  2.17s/it]
Epoch 1/25 : Average Loss 13.928
Iter (loss=3.276): : 30091it [18:13:48,  2.18s/it]
Epoch 2/25 : Average Loss 5.549
Iter (loss=4.163): : 7380it [4:29:38,  2.19s/it]
...
```

