export DATA_FILE=/content/pytorchic-bert/data/birds_corpus.txt
export BERT_PRETRAIN=/content/pytorchic-bert/pretrain
export SAVE_DIR=/content/pytorchic-bert/output
#save_dir='../exp/bert/pretrain',
#log_dir='../exp/bert/pretrain/runs',

python pretrain.py \
    --train_cfg config/pretrain.json \
    --model_cfg config/bert_base.json \
    --data_file $DATA_FILE \
    --vocab $BERT_PRETRAIN/vocab.txt \
    --save_dir $SAVE_DIR \
    --log_dir $SAVE_DIR/runs \
    --max_len 64 \
    --max_pred 20 \
    --mask_prob 0.15