# Neural Machine Translation

## Pre-trained models

Description | Dataset | Model | Test set(s)
---|---|---|---
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2) <br> newstest2012/2013: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.ntst1213.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT14 English-German](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-de.newstest2014.tar.bz2)
Convolutional <br> ([Gehring et al., 2017](https://arxiv.org/abs/1705.03122)) | [WMT17 English-German](http://statmt.org/wmt17/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt17.v2.en-de.fconv-py.tar.bz2) | newstest2014: <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt17.v2.en-de.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT14 English-French](http://statmt.org/wmt14/translation-task.html#Download) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt14.en-fr.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Ott et al., 2018](https://arxiv.org/abs/1806.00187)) | [WMT16 English-German](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2) | newstest2014 (shared vocab): <br> [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/data/wmt16.en-de.joined-dict.newstest2014.tar.bz2)
Transformer <br> ([Edunov et al., 2018](https://arxiv.org/abs/1808.09381); WMT'18 winner) | [WMT'18 English-German](http://www.statmt.org/wmt18/translation-task.html) | [download (.tar.bz2)](https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.bz2) | See NOTE in the archive

## Example usage

Generation with the binarized test sets can be run in batch mode as follows, e.g. for WMT 2014 English-French on a GTX-1080ti:
```
$ mkdir -p data-bin
$ curl https://dl.fbaipublicfiles.com/fairseq/models/wmt14.v2.en-fr.fconv-py.tar.bz2 | tar xvjf - -C data-bin
$ curl https://dl.fbaipublicfiles.com/fairseq/data/wmt14.v2.en-fr.newstest2014.tar.bz2 | tar xvjf - -C data-bin
$ fairseq-generate data-bin/wmt14.en-fr.newstest2014  \
  --path data-bin/wmt14.en-fr.fconv-py/model.pt \
  --beam 5 --batch-size 128 --remove-bpe | tee /tmp/gen.out
...
| Translated 3003 sentences (96311 tokens) in 166.0s (580.04 tokens/s)
| Generate test with beam=5: BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)

# Compute BLEU score
$ grep ^H /tmp/gen.out | cut -f3- > /tmp/gen.out.sys
$ grep ^T /tmp/gen.out | cut -f2- > /tmp/gen.out.ref
$ fairseq-score --sys /tmp/gen.out.sys --ref /tmp/gen.out.ref
BLEU4 = 40.83, 67.5/46.9/34.4/25.5 (BP=1.000, ratio=1.006, syslen=83262, reflen=82787)
```

## Preprocessing

These scripts provide an example of pre-processing data for the NMT task.

### prepare-iwslt14.sh

Provides an example of pre-processing for IWSLT'14 German to English translation task: ["Report on the 11th IWSLT evaluation campaign" by Cettolo et al.](http://workshop2014.iwslt.org/downloads/proceeding.pdf)

Example usage:
```
$ cd examples/translation/
$ bash prepare-iwslt14.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/iwslt14.tokenized.de-en
$ fairseq-preprocess --source-lang de --target-lang en --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test --destdir data-bin/iwslt14.tokenized.de-en

# Train the model (better for a single GPU setup):
$ mkdir -p checkpoints/fconv
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en --lr 0.25 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler fixed --force-anneal 200 --arch fconv_iwslt_de_en --save-dir checkpoints/fconv

# Generate:
$ fairseq-generate data-bin/iwslt14.tokenized.de-en --path checkpoints/fconv/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe

```
/home/lothar/Documents/cpsbs/checkpoints/fconv
/home/lothar/Documents/cpsbs/examples/translation/iwslt14.tokenized.de-en

python generate.py --cps --num-experiments 3 --beam 5 --nbest 5 --nucleus-threshold 0.99 --unnormalized --sampling-temperature 0.1 data-bin/iwslt14.tokenized.de-en --path checkpoints/fconv/checkpoint_best.pt


CUDA_VISIBLE_DEVICES=2 fairseq-generate data-bin/wmt14.en-fr.joined-dict.newstest2014 --path checkpoints/wmt14.en-fr.joined-dict.transformer/model.pt --cps --num-experiments 3 --beam 5 --nbest 5 --nucleus-threshold 0.99 --unnormalized --sampling-temperature 0.8


To train transformer model on IWSLT'14 German to English:
```
# Preparation steps are the same as for fconv model.

# Train the model (better for a single GPU setup):
$ mkdir -p checkpoints/transformer
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt14.tokenized.de-en \
  -a transformer_iwslt_de_en --optimizer adam --lr 0.0005 -s de -t en \
  --label-smoothing 0.1 --dropout 0.3 --max-tokens 4000 \
  --min-lr '1e-09' --lr-scheduler inverse_sqrt --weight-decay 0.0001 \
  --criterion label_smoothed_cross_entropy --max-update 50000 \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --adam-betas '(0.9, 0.98)' --save-dir checkpoints/transformer

# Average 10 latest checkpoints:
$ python scripts/average_checkpoints.py --inputs checkpoints/transformer \
   --num-epoch-checkpoints 10 --output checkpoints/transformer/model.pt

# Generate:
$ fairseq-generate data-bin/iwslt14.tokenized.de-en \
  --path checkpoints/transformer/model.pt \
  --batch-size 128 --beam 5 --remove-bpe

```

### prepare-wmt14en2de.sh

The WMT English to German dataset can be preprocessed using the `prepare-wmt14en2de.sh` script.
By default it will produce a dataset that was modeled after ["Attention Is All You Need" (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762), but with news-commentary-v12 data from WMT'17.

To use only data available in WMT'14 or to replicate results obtained in the original ["Convolutional Sequence to Sequence Learning" (Gehring et al., 2017)](https://arxiv.org/abs/1705.03122) paper, please use the `--icml17` option.

```
$ bash prepare-wmt14en2de.sh --icml17
```

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2de.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/wmt17_en_de
$ fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt17_en_de --thresholdtgt 0 --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1500 instead
$ mkdir -p checkpoints/fconv_wmt_en_de
$ fairseq-train data-bin/wmt17_en_de \
  --lr 0.5 --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_de --save-dir checkpoints/fconv_wmt_en_de

# Generate:
$ fairseq-generate data-bin/wmt17_en_de \
  --path checkpoints/fconv_wmt_en_de/checkpoint_best.pt --beam 5 --remove-bpe

```

### prepare-wmt14en2fr.sh

Provides an example of pre-processing for the WMT'14 English to French translation task.

Example usage:

```
$ cd examples/translation/
$ bash prepare-wmt14en2fr.sh
$ cd ../..

# Binarize the dataset:
$ TEXT=examples/translation/wmt14_en_fr
$ fairseq-preprocess --source-lang en --target-lang fr \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/wmt14_en_fr --thresholdtgt 0 --thresholdsrc 0

# Train the model:
# If it runs out of memory, try to set --max-tokens 1000 instead
$ mkdir -p checkpoints/fconv_wmt_en_fr
$ fairseq-train data-bin/wmt14_en_fr \
  --lr 0.5 --clip-norm 0.1 --dropout 0.1 --max-tokens 3000 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --lr-scheduler fixed --force-anneal 50 \
  --arch fconv_wmt_en_fr --save-dir checkpoints/fconv_wmt_en_fr

# Generate:
$ fairseq-generate data-bin/fconv_wmt_en_fr \
  --path checkpoints/fconv_wmt_en_fr/checkpoint_best.pt --beam 5 --remove-bpe

```

## Multilingual Translation

We also support training multilingual translation models. In this example we'll
train a multilingual `{de,fr}-en` translation model using the IWSLT'17 datasets.

Note that we use slightly different preprocessing here than for the IWSLT'14
En-De data above. In particular we learn a joint BPE code for all three
languages and use interactive.py and sacrebleu for scoring the test set.

```
# First install sacrebleu and sentencepiece
$ pip install sacrebleu sentencepiece

# Then download and preprocess the data
$ cd examples/translation/
$ bash prepare-iwslt17-multilingual.sh
$ cd ../..

# Binarize the de-en dataset
$ TEXT=examples/translation/iwslt17.de_fr.en.bpe16k
$ fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/train.bpe.de-en --validpref $TEXT/valid.bpe.de-en \
  --joined-dictionary \
  --destdir data-bin/iwslt17.de_fr.en.bpe16k \
  --workers 10

# Binarize the fr-en dataset
# NOTE: it's important to reuse the en dictionary from the previous step
$ fairseq-preprocess --source-lang fr --target-lang en \
  --trainpref $TEXT/train.bpe.fr-en --validpref $TEXT/valid.bpe.fr-en \
  --joined-dictionary --tgtdict data-bin/iwslt17.de_fr.en.bpe16k/dict.en.txt \
  --destdir data-bin/iwslt17.de_fr.en.bpe16k \
  --workers 10

# Train a multilingual transformer model
# NOTE: the command below assumes 1 GPU, but accumulates gradients from
#       8 fwd/bwd passes to simulate training on 8 GPUs
$ mkdir -p checkpoints/multilingual_transformer
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
  --max-epoch 50 \
  --ddp-backend=no_c10d \
  --task multilingual_translation --lang-pairs de-en,fr-en \
  --arch multilingual_transformer_iwslt_de_en \
  --share-decoders --share-decoder-input-output-embed \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.0005 --lr-scheduler inverse_sqrt --min-lr '1e-09' \
  --warmup-updates 4000 --warmup-init-lr '1e-07' \
  --label-smoothing 0.1 --criterion label_smoothed_cross_entropy \
  --dropout 0.3 --weight-decay 0.0001 \
  --save-dir checkpoints/multilingual_transformer \
  --max-tokens 4000 \
  --update-freq 8

# Generate and score the test set with sacrebleu
$ SRC=de
$ sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src \
  | python scripts/spm_encode.py --model examples/translation/iwslt17.de_fr.en.bpe16k/sentencepiece.bpe.model \
  > iwslt17.test.${SRC}-en.${SRC}.bpe
$ cat iwslt17.test.${SRC}-en.${SRC}.bpe | fairseq-interactive data-bin/iwslt17.de_fr.en.bpe16k/ \
  --task multilingual_translation --source-lang ${SRC} --target-lang en \
  --path checkpoints/multilingual_transformer/checkpoint_best.pt \
  --buffer 2000 --batch-size 128 \
  --beam 5 --remove-bpe=sentencepiece \
  > iwslt17.test.${SRC}-en.en.sys
$ grep ^H iwslt17.test.${SRC}-en.en.sys | cut -f3 \
  | sacrebleu --test-set iwslt17 --language-pair ${SRC}-en

  import numpy as np
from libc.math cimport NAN

cimport cython
from cython.parallel import prange

cimport numpy as np
from libc.math cimport exp, log1p, log, expm1, abs


cdef extern from "math.h":
    float INFINITY

ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE_int_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t log1mexp(DTYPE_t x):
    """
    Numerically stable implementation of log(1-exp(x))
    Note: function is finite for x < 0.
    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    cdef DTYPE_t a
    if x >= 0:
        return NAN
    else:
        a = abs(x)
        if 0 < a <= 0.693:
            return log(-expm1(-a))
        else:
            return log1p(-exp(-a))

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t log1pexp(DTYPE_t x) nogil:
    """
    Numerically stable implementation of log(1+exp(x)) aka softmax(0,x).
    -log1pexp(-x) is log(sigmoid(x))
    Source:
    http://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    if x <= -37.:
        return exp(x)
    elif -37. <= x <= 18.:
        return log1p(exp(x))
    elif 18. < x <= 33.3:
        return x + exp(-x)
    else:
        return x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline DTYPE_t log_add(DTYPE_t x, DTYPE_t y) nogil:
    """
    Addition of 2 values in log space.
    Need separate checks for inf because inf-inf=nan
    """
    if x == -INFINITY:
        return y
    elif y == -INFINITY:
        return x
    else:
        if y <= x:
            return x + log1pexp(y - x)
        else:
            return y + log1pexp(x - y)

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_normalization(np.ndarray[DTYPE_t, ndim=1] logp_sliced, int k):
    """
    This function calculates the normalization factor in CPS which is
    sum of product of weights of all sets that have size k
    @param logp_sliced: weights of candidates in log space
    @param k: sample size
    @return: dp matrix containing all normalization factors
    """
    cdef int n = len(logp_sliced)
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs

    subset_sum_product_probs = np.full((k + 1, n + 1), -np.inf, dtype=np.float64)
    subset_sum_product_probs[0, :] = 0.
    cdef float intermediate_res
    cdef int r
    cdef int i

    for r in range(1, k + 1):
        for i in prange(1, n + 1, nogil=True):
            intermediate_res = subset_sum_product_probs[r - 1, i - 1] + logp_sliced[i - 1]
            subset_sum_product_probs[r, i] = log_add(subset_sum_product_probs[r, i - 1], intermediate_res)
    return subset_sum_product_probs

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_log_inclusion_probs(np.ndarray[DTYPE_t, ndim=1] logp_sliced,
                             np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs, int k):
    """
    This function calculates the inclusion probability for CPS design
    operates in log space
    @param logp_sliced: weights of candidates which can be probabilities or odds
    @param subset_sum_product_probs: normalization factors
    @param k:sample size
    @return: log inclusion probabilities
    """
    cdef int n = len(logp_sliced)
    cdef np.ndarray[DTYPE_t, ndim=1] dp = np.full(n, -np.inf, dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs

    cdef np.ndarray[DTYPE_t, ndim=2] remaining_subsetsum_product_probs = np.full((k + 2, n + 2), -np.inf,
                                                                                 dtype=np.float64)
    remaining_subsetsum_product_probs[k, :] = 0.

    cdef int r
    cdef int i
    for r in range(k, 0, -1):
        for i in prange(n, 0, -1, nogil=True):
            dp[i - 1] = log_add(dp[i - 1],
                                subset_sum_product_probs[r - 1, i - 1] + remaining_subsetsum_product_probs[r, i + 1])
            remaining_subsetsum_product_probs[r, i] = log_add(
                remaining_subsetsum_product_probs[r + 1, i + 1] + logp_sliced[i - 1],
                remaining_subsetsum_product_probs[r, i + 1])

    log_inclusion_probs = logp_sliced + dp - subset_sum_product_probs[k, n]
    return log_inclusion_probs

@cython.boundscheck(False)
@cython.wraparound(False)
def sample(np.ndarray[DTYPE_t, ndim=1] logp, np.ndarray[DTYPE_int_t, ndim=1] selected_inds, int k):
    """
    This function picks a sample of size k from candidates
    @param logp: log probability of candidates
    @param selected_inds: selected candidates after nucleus filtering
    @param k: sample size
    @return: selected candidates indices and their inclusion probabilities
    """
    cdef long n = len(logp)
    k = min(n, k)

    cdef list samples_idx = []
    cdef list selected_incs = []
    cdef np.ndarray[DTYPE_t, ndim=1] thresholds = np.log(np.random.uniform(size=n))

    cdef np.ndarray[DTYPE_t, ndim=1] log_weights # using odds approximation as weights
    cdef np.ndarray[DTYPE_t, ndim=1] log_prob_filtered
    log_prob_filtered = logp.copy()
    log_prob_filtered[log_prob_filtered > 0.99] = 0.99 # clipping in order to prevent NAN generation
    log_weights = log_prob_filtered - np.array(list(map(log1mexp, log_prob_filtered)))  

    cdef long i
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs
    cdef DTYPE_t thresh
    cdef int to_pick_number
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs
    to_pick_number = k
    subset_sum_product_probs = calc_normalization(log_weights, k)
    log_inclusion_probs = calc_log_inclusion_probs(log_weights, subset_sum_product_probs, k)
    for i in range(n, 0, -1):
        thresh = log_weights[i - 1] + subset_sum_product_probs[to_pick_number - 1, i - 1] - subset_sum_product_probs[
            to_pick_number, i]
        if thresholds[i - 1] < thresh:
            samples_idx.append(selected_inds[i - 1])
            selected_incs.append(log_inclusion_probs[i - 1])
            to_pick_number -= 1
            if to_pick_number == 0:
                break
    return np.asarray(samples_idx), np.asarray(selected_incs)


@cython.boundscheck(False)
@cython.wraparound(False)
def sampford_sample(np.ndarray[DTYPE_t, ndim=1] logp, np.ndarray[DTYPE_int_t, ndim=1] selected_inds, int k):
    cdef long n = len(logp)
    k = min(n, k)


    cdef np.ndarray[DTYPE_t, ndim=1] thresholds = np.log(np.random.uniform(size=n))

    cdef np.ndarray[DTYPE_t, ndim=1] log_weights # using odds approximation as weights
    cdef np.ndarray[DTYPE_t, ndim=1] log_prob_filtered
    log_prob_filtered = logp.copy()
    log_prob_filtered[log_prob_filtered > 0.99] = 0.99 # clipping in order to prevent NAN generation
    log_weights = log_prob_filtered - np.array(list(map(log1mexp, log_prob_filtered)))  

    cdef long i
    cdef np.ndarray[DTYPE_t, ndim=2] subset_sum_product_probs
    cdef DTYPE_t thresh
    cdef int to_pick_number
    cdef np.ndarray[DTYPE_t, ndim=1] log_inclusion_probs
    to_pick_number = k
    subset_sum_product_probs = calc_normalization(log_weights, k)
    log_inclusion_probs = calc_log_inclusion_probs(log_weights, subset_sum_product_probs, k)
    inclusion_probs = log_inclusion_probs.copy()
    #inclusion_probs[inclusion_probs<-100000] = 0.0
    inclusion_probs = np.exp(inclusion_probs)
    print(inclusion_probs)
    inclusion_probs = inclusion_probs/np.linalg.norm(inclusion_probs,ord=1)
    print(len(selected_inds))
    print(inclusion_probs)

    #einmal samplen mit den wahrscheinlichkeiten inclusion probability/n
    cdef int j = np.random.choice(selected_inds, 1, p=inclusion_probs)
    
    
    #verwende die cps methode mit neuen parametern
    samples_idx, selected_incs =  sample(log_inclusion_probs, selected_inds, k-1)
    
    #b sagt aus, ob im neuen cps iteration das ursprüngliche j wieder drinne ist
    cdef int b = 0
    if j in samples_idx:
        b = 1

    
    #Falls das j jetzt "doppelt" gesampelt worden ist, widerholen bis nicht
    while b==1:
        samples_idx, selected_incs =  sample(log_inclusion_probs, selected_inds, k-1)
        if j in samples_idx:
            b = 1
        else:
            b = 0
    
    #Füge dein j doch noch hinzu
    #Reihenfolge egal?

    np.append(samples_idx,j)
    np.append(selected_incs,log_inclusion_probs[j])
    samples_idx = samples_idx.astype(int)
    print(type(samples_idx))

    return np.asarray(samples_idx), np.asarray(selected_incs)
```
