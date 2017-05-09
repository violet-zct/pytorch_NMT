from __future__ import division
from collections import defaultdict, Counter
import numpy as np


def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_slice(data, batch_size, sort=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in xrange(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        src_sents = [data[i * batch_size + b][0] for b in range(cur_batch_size)]
        tgt_sents = [data[i * batch_size + b][1] for b in range(cur_batch_size)]

        if sort:
            src_ids = sorted(range(cur_batch_size), key=lambda src_id: len(src_sents[src_id]), reverse=True)
            src_sents = [src_sents[src_id] for src_id in src_ids]
            tgt_sents = [tgt_sents[src_id] for src_id in src_ids]

        yield src_sents, tgt_sents


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of source sentences in each batch is decreasing
    """

    buckets = defaultdict(list)
    for pair in data:
        src_sent = pair[0]
        buckets[len(src_sent)].append(pair)

    batched_data = []
    for src_len in buckets:
        tuples = buckets[src_len]
        if shuffle: np.random.shuffle(tuples)
        batched_data.extend(list(batch_slice(tuples, batch_size)))

    if shuffle:
        np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch

def ngrams(sequence, n):
    """
    borrowed from NLTK
    """

    sequence = iter(sequence)
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]

def get_length(reference, hypothesis):
    difference = abs(len(reference) - len(hypothesis))
    difference = difference*1.0/ len(reference)
    return -difference

def get_repeat(reference, hypothesis):
    previous = hypothesis[0]
    count = 0.
    for word in hypothesis[1:]:
        if word == previous:
            count+=1
        previous = word
    return -count*1.0/len(hypothesis)



def calc_f1(reference, hypothesis):
    """
    F1 score between reference and hypothesis
    """
    f1_scores = []
    k = 1
    for n in xrange(1, 5):
        hyp_ngram_counts = Counter(ngrams(hypothesis, n))
        ref_ngram_counts = Counter(ngrams(reference, n))

        ngram_f1 = 0.
        if len(reference) < n:
            continue

        if len(hypothesis) >= n and len(reference) >= n:
            ngram_prec = sum(ngram_count for ngram, ngram_count in hyp_ngram_counts.iteritems() if ngram in ref_ngram_counts) / (len(hypothesis) - n + 1)
            ngram_recall = sum(ngram_count for ngram, ngram_count in ref_ngram_counts.iteritems() if ngram in hyp_ngram_counts) / (len(reference) - n + 1)

            ngram_f1 = 2 * ngram_prec * ngram_recall / (ngram_prec + ngram_recall) if ngram_prec + ngram_recall > 0. else 0.

        if len(hypothesis) < n <= len(reference) or ngram_f1 == 0.:
            ngram_f1 = 1. / (2 ** k * (len(reference) - n + 1))

        f1_scores.append(ngram_f1)

    f1_mean = (np.prod(f1_scores)) ** (1 / len(f1_scores))

    return f1_mean

if __name__ == '__main__':
    ref_sent = 'thank you .'.split(' ')
    hyp_sent = 'thank'.split(' ')

    print calc_f1(ref_sent, hyp_sent)