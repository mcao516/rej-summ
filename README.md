# Learning with Rejection for Abstractive Text Summarization
This directory contains code necessary to replicate the training and evaluation for our EMNLP 2022 paper "[Learning with Rejection for Abstractive Text Summarization](https://aclanthology.org/2022.emnlp-main.663/)" by [Meng Cao](https://mcao516.github.io/), [Yue Dong](https://yuedongcs.github.io/), [Jingyi He](https://kylie-box.github.io/) and [Jackie Chi Kit Cheung](https://www.cs.mcgill.ca/~jcheung/).

Our implementation is heavily based on facebook's [fairseq](https://github.com/facebookresearch/fairseq) library. The core implementation of the algorithm is in the ```fairseq/criterions/label_smoothed_cross_entropy_with_rejection.py``` file.

# Requirements and Installation
* [PyTorch](http://pytorch.org/) version >= 1.10.0
* Python version >= 3.8
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* To install and develop locally:

``` bash
git clone https://github.com/mcao516/rej-summ.git
cd rej-summ
pip install --editable ./
```

# Running the Code
To reproduce the results in the paper, you can download the preprocessed XSum dataset from google drive using this [link](https://drive.google.com/file/d/1zZrhxOAgD2qc4dMrFpXrYxHm-XC85yYY/view?usp=sharing).

## Training

``` bash
TOTAL_NUM_UPDATES=20000
WARMUP_UPDATES=500      
LR=3e-05
MAX_TOKENS=2048
UPDATE_FREQ=2

BART_PATH=${HOME}/BART_models/bart.large/model.pt
DATA_PATH=${HOME}/summarization/XSum/xsum-bin
SAVE_DIR=checkpoints/
mkdir $SAVE_DIR

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fairseq-train $DATA_PATH \
    --max-epoch 3 \
    --abstention-mask-dir ${HOME}/summarization/XSum/masks/ \
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

## Inference
```bash
DATA_PATH=${HOME}/summarization/XSum/xsum-bin
SRC_PATH=$HOME/summarization/XSum/test.source
OUTPUT_PATH=hypos/output.hypo

CUDA_VISIBLE_DEVICES=0 python examples/bart/summarize.py \
    --model-dir checkpoints/ \
    --model-file checkpoint_best.pt \
    --dict-dir $DATA_PATH \
    --src $SRC_PATH \
    --out $OUTPUT_PATH \
    --beam_size 6 \
    --bsz 8 \
    --unnormalized \
    --lenpen 1.0 \
    --rejpen 2.0 \
    --xsum-kwargs;
```

# Running the Code on Your Own Data


# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{cao-etal-2022-learning,
    title = "Learning with Rejection for Abstractive Text Summarization",
    author = "Cao, Meng  and
      Dong, Yue  and
      He, Jingyi  and
      Cheung, Jackie Chi Kit",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.663",
    pages = "9768--9780",
    abstract = "State-of-the-art abstractive summarization systems frequently hallucinate content that is not supported by the source document, mainly due to noise in the training dataset.Existing methods opt to drop the noisy samples or tokens from the training set entirely, reducing the effective training set size and creating an artificial propensity to copy words from the source. In this work, we propose a training objective for abstractive summarization based on rejection learning, in which the model learns whether or not to reject potentially noisy tokens. We further propose a regularized decoding objective that penalizes non-factual candidate summaries during inference by using the rejection probability learned during training.We show that our method considerably improves the factuality of generated summaries in automatic and human evaluations when compared to five baseline models, and that it does so while increasing the abstractiveness of the generated summaries.",
}
```
