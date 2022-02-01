![Banner](https://github.com/AfraAmini/cpsbs/blob/main/header.jpg)

# Sampford Sampling Beam Search

This repository contains implementation of Sampford Sampling, which can be used to draw samples *without* replacement from sequence models.

# Table of contents
- [Project Summary](#conditional-poisson-beams)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)

# Installation
[(Back to top)](#table-of-contents)

For detailed installation instructions and basic fairseq usage instruction, please go to [the fairseq repository](https://github.com/pytorch/fairseq).

Basically, to install and develop fairseq locally:
```bash
pip install --editable ./

```


# Usage 
[(Back to top)](#table-of-contents)

Example:
```bash
python generate.py --sampford --num-experiments 3 
                    --beam 5 --nbest 5 
                    --nucleus-threshold 0.99
                    --unnormalized --sampling-temperature 0.1 
                    [DATAPATH] --path [MODELPATH]
```
- ``--sampford``: to use Sampford for decoding
- ``-num-experiments``: repeat the procedure for the specified number of times. Useful for building estimators.
- ``--beam``: beam size or sample size
- ``--nbest``: equal to ``--beam`` (using a beam size greater than nbest is equivalent)
- ``--nucleus-threshold``: probability threshold for nucleus filtering. Default is 1. (no filtering)
- ``--no-early-stopping`` and ``--unnormalized``: for theoretical correctness of the sampling algorithm 
- ``--sampling_temperature``: temperature used for local softmax normalization. 

# Reproducibility
We used the conv.wmt14.en-fr model from Gehring et al. and the newstest2014 dataset (https://github.com/pytorch/fairseq/blob/main/examples/translation/README.md). The commands used to run our experiments can be found in run_script.sh.

Example commands for beam size 10 and sampling temperature 0.1:
```bash
#Beam Search
CUDA_VISIBLE_DEVICES=0 fairseq-generate [DATAPATH] --path [MODELPATH] --beam 10 --nbest 10 --no-early-stop --unnormalized --sampling-temperature 0.1 --batch-size 1

#Sampling
CUDA_VISIBLE_DEVICES=0 fairseq-generate --sampling [DATAPATH] --path [MODELPATH] --beam 10 --nbest 10 --sampling-temperature 0.1 --no-early-stop --unnormalized --batch-size 1 --sampling-topk 10

#Stochastic Beam Search
CUDA_VISIBLE_DEVICES=0 fairseq-generate --stochastic-beam-search [DATAPATH] --path [MODELPATH] --beam 10 --nbest 10 --sampling-temperature 0.1 --no-early-stop --unnormalized --batch-size 1

#Diverse Beam Search
CUDA_VISIBLE_DEVICES=0 fairseq-generate --diverse-beam-groups 10 [DATAPATH] --path [MODELPATH] --beam 10 --nbest 10 --diverse-beam-strength 0.1 --no-early-stop --unnormalized --batch-size 1 

#Sampford Sampling
CUDA_VISIBLE_DEVICES=0 fairseq-generate [DATAPATH] --path [MODELPATH] --sampford --beam 10 --nbest 10 --unnormalized --batch-size 1  --sampling-temperature 0.1
```

