# Learning with Rejection for Abstractive Text Summarization
This directory contains code necessary to replicate the training and evaluation for our EMNLP 2022 paper "Learning with Rejection for Abstractive Text Summarization" by Meng Cao, Yue Dong, Jingyi He and Jackie Chi Kit Cheung.

# Requirements and Installation
Our implementation is heavily based on facebook's [fairseq](https://github.com/facebookresearch/fairseq) library.

* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* To install and develop locally:

``` bash
git clone https://github.com/mcao516/rej-summ.git
cd rej-summ
pip install --editable ./
```

# Training

``` bash
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=2

BART_PATH=${HOME}/BART_models/bart.large/model.pt
DATA_PATH=${HOME}/summarization/XSum/fairseq_files/xsum-bin
SAVE_DIR=${HOME}/checkpoints/
mkdir $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train $DATA_PATH \
    --max-epoch 3 \
    --abstention-mask-dir ${HOME}/summarization/XSum/fairseq_files/masks/ \
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
```

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

<!-- ``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
``` -->
