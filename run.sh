#!/usr/bin/env bash
python nmt.py \
    --train_src ./en-de/train.en-de.low.filt.de \
    --train_tgt ./en-de/train.en-de.low.filt.en \
    --dev_src ./en-de/valid.en-de.low.de \
    --dev_tgt ./en-de/valid.en-de.low.en \
    --test_src ./en-de/test.en-de.low.de \
    --test_tgt ./en-de/test.en-de.low.en \
    --vocab ./vocab.bin \
    --model_type mixer \
    --sample_size 5 \
    --reward_type "bleu" \
    --valid_niter 1000 \
    --load_model mle.bin \
    --save_to mle_rl_bleu \
    --decode_max_time_step 100 \
    --batch_size 5 \
    --log_every 5
