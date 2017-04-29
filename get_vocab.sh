#!/usr/bin/env bash
python vocab.py \
    --train_src ./en-de/train.en-de.low.filt.de \
    --train_tgt ./en-de/train.en-de.low.filt.en \
    --src_vocab_size 30000 \
    --tgt_vocab_size 20000
