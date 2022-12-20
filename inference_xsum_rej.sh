#!/bin/bash
module load libffi
source $HOME/env38/bin/activate

CUDA_VISIBLE_DEVICES=1 python examples/bart/summarize.py \
    --model-dir $SCRATCH/BART_models/abstention_entonly_alpha1_mila-test_rewrite-loss_separate-loss-file_epoch_3/ \
    --model-file checkpoint_best.pt \
    --dict-dir $SCRATCH/summarization/XSum/fairseq_files/xsum-bin \
    --src $SCRATCH/summarization/XSum/fairseq_files/test.source \
    --out hypos/xsum_test_rej_alpha-1_lambda-2_rewrite-loss_separate-loss-file_epoch_3_unnormalized.hypo \
    --beam_size 6 \
    --bsz 8 \
    --unnormalized \
    --lenpen 1.0 \
    --rejpen 2.0 \
    --xsum-kwargs;
