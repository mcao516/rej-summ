#!/bin/bash

# load your virtual environment
module load libffi
source $HOME/env38/bin/activate

TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=2

BART_PATH=$SCRATCH/BART_models/bart.large/model.pt
DATA_PATH=$SCRATCH/summarization/XSum/fairseq_files/xsum-bin
SAVE_DIR=$SCRATCH/BART_models/abstention_entonly_alpha1_mila-test_rewrite-loss_separate-loss-file_new_mask_epoch_3/
mkdir $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train $DATA_PATH \
    --max-epoch 3 \
    --abstention-mask-dir $HOME/rej-summ/masks/ \
    --rejection-alpha 1.0 \
    --restore-file $BART_PATH \
    --save-dir $SAVE_DIR \
    --max-tokens $MAX_TOKENS \
    --task translation \
    --source-lang source --target-lang target \
    --truncate-source \
    --layernorm-embedding \
    --share-all-embeddings \
    --share-decoder-input-output-embed \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch bart_large \
    --criterion label_smoothed_cross_entropy_with_rejection \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
    --clip-norm 0.1 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --fp16 --update-freq $UPDATE_FREQ \
    --skip-invalid-size-inputs-valid-test \
    --find-unused-parameters;
