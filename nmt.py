from __future__ import print_function

import re

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
from torch import optim
from torch.nn import Parameter
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu,SmoothingFunction
import time
import numpy as np
from collections import defaultdict, Counter, namedtuple
from itertools import chain, islice
import argparse, os, sys

from util import *
from vocab import Vocab, VocabEntry


def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=5783287, type=int, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False, help='use gpu')
    parser.add_argument('--mode', choices=['train', 'test', 'sample'], default='train', help='run mode')
    parser.add_argument('--vocab', type=str, help='path of the serialized vocabulary')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--beam_size', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--sample_size', default=5, type=int, help='sample size')
    parser.add_argument('--embed_size', default=256, type=int, help='size of word embeddings')
    parser.add_argument('--hidden_size', default=256, type=int, help='size of LSTM hidden states')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate')

    parser.add_argument('--train_src', type=str, help='path to the training source file')
    parser.add_argument('--train_tgt', type=str, help='path to the training target file')
    parser.add_argument('--dev_src', type=str, help='path to the dev source file')
    parser.add_argument('--dev_tgt', type=str, help='path to the dev target file')
    parser.add_argument('--test_src', type=str, help='path to the test source file')
    parser.add_argument('--test_tgt', type=str, help='path to the test target file')

    parser.add_argument('--decode_max_time_step', default=200, type=int, help='maximum number of time steps used '
                                                                              'in decoding and sampling')

    parser.add_argument('--model_type', default='ml', type=str, choices=['ml','rl', 'mixer'])
    parser.add_argument('--valid_niter', default=500, type=int, help='every n iterations to perform validation')
    parser.add_argument('--valid_metric', default='bleu', choices=['bleu', 'ppl', 'word_acc', 'sent_acc'], help='metric used for validation')
    parser.add_argument('--log_every', default=500, type=int, help='every n iterations to log training statistics')
    parser.add_argument('--load_model', default=None, type=str, help='load a pre-trained model')
    parser.add_argument('--save_to', default='model', type=str, help='save trained model to')
    parser.add_argument('--save_model_after', default=2, help='save the model only after n validation iterations')
    parser.add_argument('--save_to_file', default=None, type=str, help='if provided, save decoding results to file')
    parser.add_argument('--save_nbest', default=False, action='store_true', help='save nbest decoding results')
    parser.add_argument('--patience', default=30, type=int, help='training patience')
    parser.add_argument('--uniform_init', default=None, type=float, help='if specified, use uniform initialization for all parameters')
    parser.add_argument('--clip_grad', default=5., type=float, help='clip gradients')
    parser.add_argument('--max_niter', default=-1, type=int, help='maximum number of training iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=0.5, type=float, help='decay learning rate if the validation performance drops')
    parser.add_argument('--update_freq', default=1, type=int, help='update freq')
    parser.add_argument('--reward_type', default='bleu', type=str, choices=['bleu', 'f1', 'combined','delta_f1'])

    parser.add_argument('--run_with_no_patience', default=10, type=int, help="If patience hit within these many first epochs, reset patience=0.")
    parser.add_argument('--delta_steps', default=3, type=int, help='MIXER: annealing steps for using REINFROCE loss')
    parser.add_argument('--XER', default=2, type=int, help='Every XER epoches, decrease delta by delta.' )
    # raml training
    # parser.add_argument('--temp', default=0.85, type=float, help='temperature in reward distribution')
    # parser.add_argument('--raml_sample_file', type=str, help='path to the sampled targets')

    args = parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed * 13 / 7)

    return args


def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    masks = []
    for i in xrange(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in xrange(batch_size)])
        masks.append([1 if len(sents[k]) > i else 0 for k in xrange(batch_size)])

    return sents_t, masks


def word2id(sents, vocab):
    if type(sents[0]) == list:
        return [[vocab[w] for w in s] for s in sents]
    else:
        return [vocab[w] for w in sents]


def tensor_transform(linear, X):
    # X is a 3D tensor
    return linear(X.contiguous().view(-1, X.size(2))).view(X.size(0), X.size(1), -1)


def get_reward(tgt_sent, sample, reward='bleu', max_len=0):
    sm = SmoothingFunction()
    if reward=='bleu':
        score = sentence_bleu([tgt_sent], sample, smoothing_function=sm.method3)
    elif reward == 'f1':
        score = calc_f1(tgt_sent, sample)
    elif reward == 'combined':
        score = sentence_bleu([tgt_sent], sample, smoothing_function=sm.method3) + calc_f1(tgt_sent, sample)
    elif reward =='delta_f1':
        score_list = []
        prev_score = 0.
        delta_scores = []
        for l in xrange(1, len(sample) + 1):
            partial_hyp = sample[:l]
            y_t = sample[l - 1]
            score = calc_f1(tgt_sent, partial_hyp)
            delta_score = score - prev_score
            delta_scores.append(delta_score)
            prev_score = score

        cum_reward = 0.
        for i in reversed(xrange(len(sample))):
            reward_i = delta_scores[i]
            cum_reward += reward_i
            score_list.append(cum_reward)

        score_list = list(reversed(score_list))     
        for i in range(max_len-len(score_list)):
            score_list.append(0)
        score = score_list
    return score


class NMT(nn.Module):
    def __init__(self, args, vocab):
        super(NMT, self).__init__()

        self.args = args

        self.vocab = vocab

        self.src_embed = nn.Embedding(len(vocab.src), args.embed_size, padding_idx=vocab.src['<pad>'])
        self.tgt_embed = nn.Embedding(len(vocab.tgt), args.embed_size, padding_idx=vocab.tgt['<pad>'])

        self.encoder_lstm = nn.LSTM(args.embed_size, args.hidden_size, bidirectional=True, dropout=args.dropout)
        self.decoder_lstm = nn.LSTMCell(args.embed_size + args.hidden_size, args.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(args.hidden_size * 3, args.hidden_size, bias=False)

        # prediction layer of the target vocabulary
        self.readout = nn.Linear(args.hidden_size, len(vocab.tgt), bias=False)

        # dropout layer
        self.dropout = nn.Dropout(args.dropout)

        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(args.hidden_size * 2, args.hidden_size)

        if args.model_type == "rl" or args.model_type == "mixer":
            self.baseline = nn.Linear(args.hidden_size, 1, bias=True)

    def forward(self, src_sents, src_sents_len, tgt_words):
        src_encodings, init_ctx_vec = self.encode(src_sents, src_sents_len)
        scores = self.decode(src_encodings, init_ctx_vec, tgt_words)

        return scores

    def load_state_dict(self, state_dict):
        """Copies parameters and buffers from :attr:`state_dict` into
        this module and its descendants. The keys of :attr:`state_dict` must
        exactly match the keys returned by this module's :func:`state_dict()`
        function.

        Arguments:
            state_dict (dict): A dict containing parameters and
                persistent buffers.
        """
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state and name.startswith('baseline'):
                print("no baseline in MLE")
                continue
            elif name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            # print("Copying %", name)
            # print(own_state[name].size(), param.size())
            own_state[name].copy_(param)


        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) == 2:
            print("miss dealine parameters is fine!")
        elif len(missing)!=0:
            raise KeyError('missing keys in state_dict: "{}"'.format(missing))

    def encode(self, src_sents, src_sents_len):
        """
        :param src_sents: (src_sent_len, batch_size), sorted by the length of the source
        :param src_sents_len: (src_sent_len)
        """
        # (src_sent_len, batch_size, embed_size)
        src_word_embed = self.src_embed(src_sents)
        packed_src_embed = pack_padded_sequence(src_word_embed, src_sents_len)

        # output: (src_sent_len, batch_size, hidden_size)
        output, (last_state, last_cell) = self.encoder_lstm(packed_src_embed)
        output, _ = pad_packed_sequence(output)

        dec_init_cell = self.decoder_cell_init(torch.cat([last_cell[0], last_cell[1]], 1))
        dec_init_state = F.tanh(dec_init_cell)

        return output, (dec_init_state, dec_init_cell)

    def decode(self, src_encoding, dec_init_vec, tgt_sents):
        """
        :param src_encoding: (src_sent_len, batch_size, hidden_size)
        :param dec_init_vec: (batch_size, hidden_size)
        :param tgt_sents: (tgt_sent_len, batch_size)
        :return:
        """
        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        new_tensor = init_cell.data.new
        batch_size = src_encoding.size(1)

        # (batch_size, src_sent_len, hidden_size)
        src_encoding = src_encoding.t()
        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        # initialize attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), requires_grad=False)

        tgt_word_embed = self.tgt_embed(tgt_sents)
        scores = []

        # start from `<s>`, until y_{T-1}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            # input feeding: concate y_tm1 and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(0), att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))   # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)   # E.q. (6)
            scores.append(score_t)

            att_tm1 = att_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)
        return scores


    def translate(self, src_sents, beam_size=None, to_word=True):
        """
        perform beam search
        TODO: batched beam search
        """
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not beam_size:
            beam_size = args.beam_size

        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=args.cuda, is_test=True)

        src_encoding, dec_init_vec = self.encode(src_sents_var, [len(src_sents[0])])
        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)

        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        att_tm1 = Variable(torch.zeros(1, self.args.hidden_size), volatile=True)
        hyp_scores = Variable(torch.zeros(1), volatile=True)
        if args.cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()

        eos_id = self.vocab.tgt['</s>']
        bos_id = self.vocab.tgt['<s>']
        tgt_vocab_size = len(self.vocab.tgt)

        hypotheses = [[bos_id]]
        completed_hypotheses = []
        completed_hypothesis_scores = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < args.decode_max_time_step:
            t += 1
            hyp_num = len(hypotheses)

            expanded_src_encoding = src_encoding.expand(src_encoding.size(0), hyp_num, src_encoding.size(2))
            expanded_src_encoding_att_linear = src_encoding_att_linear.expand(src_encoding_att_linear.size(0), hyp_num,
                                                                              src_encoding_att_linear.size(2))

            y_tm1 = Variable(torch.LongTensor([hyp[-1] for hyp in hypotheses]), volatile=True)
            if args.cuda:
                y_tm1 = y_tm1.cuda()

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (hyp_num, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, expanded_src_encoding.t(), expanded_src_encoding_att_linear.t())

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)
            p_t = F.log_softmax(score_t)

            live_hyp_num = beam_size - len(completed_hypotheses)
            new_hyp_scores = (hyp_scores.unsqueeze(1).expand_as(p_t) + p_t).view(-1)
            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores, k=live_hyp_num)
            prev_hyp_ids = top_new_hyp_pos / tgt_vocab_size
            word_ids = top_new_hyp_pos % tgt_vocab_size
            # new_hyp_scores = new_hyp_scores[top_new_hyp_pos.data]

            new_hypotheses = []

            live_hyp_ids = []
            new_hyp_scores = []
            for prev_hyp_id, word_id, new_hyp_score in zip(prev_hyp_ids.cpu().data, word_ids.cpu().data, top_new_hyp_scores.cpu().data):
                hyp_tgt_words = hypotheses[prev_hyp_id] + [word_id]
                if word_id == eos_id:
                    completed_hypotheses.append(hyp_tgt_words)
                    completed_hypothesis_scores.append(new_hyp_score)
                else:
                    new_hypotheses.append(hyp_tgt_words)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.LongTensor(live_hyp_ids)
            if args.cuda:
                live_hyp_ids = live_hyp_ids.cuda()

            hidden = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hyp_scores = Variable(torch.FloatTensor(new_hyp_scores), volatile=True) # new_hyp_scores[live_hyp_ids]
            if args.cuda:
                hyp_scores = hyp_scores.cuda()
            hypotheses = new_hypotheses

        if len(completed_hypotheses) == 0:
            completed_hypotheses = [hypotheses[0]]
            completed_hypothesis_scores = [0.0]

        if to_word:
            for i, hyp in enumerate(completed_hypotheses):
                completed_hypotheses[i] = [self.vocab.tgt.id2word[w] for w in hyp]

        ranked_hypotheses = sorted(zip(completed_hypotheses, completed_hypothesis_scores), key=lambda x: x[1], reverse=True)

        return [hyp for hyp, score in ranked_hypotheses]

    def mixer(self, src_encoding, dec_init_vec, tgt_sents_var, delta, tgt_sents_tokens, cross_entropy_loss, reward_type, to_word=False):
        """
        :param src_encoding: (src_sent_len, batch_size, hidden_size)
        :param dec_init_vec: (batch_size, hidden_size)
        :param tgt_sents: (tgt_sent_len, batch_size)
        :return:
        """

        # mixer is called when delta < T
        # tgt_sents: 0 : -(delta+2)
        # ref_tgt_sents: 1: -(delta+1)
        init_state = dec_init_vec[0]
        init_cell = dec_init_vec[1]
        hidden = (init_state, init_cell)

        new_tensor = init_cell.data.new
        batch_size = src_encoding.size(1)

        # (batch_size, src_sent_len, hidden_size)
        src_encoding = src_encoding.t()
        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        # initialize attentional vector
        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_(), requires_grad=False)

        tgt_word_embed = self.tgt_embed(tgt_sents_var[:-(delta+1)])
        scores = []

        max_len = tgt_sents_var.size(0)
        CE_length = max_len - delta
        # print("Maximum length and CE length and CE embedding length: ", max_len, CE_length, tgt_word_embed.size(0))
        # TODO: start from `<s>`, until y_{T-delta}
        for y_tm1_embed in tgt_word_embed.split(split_size=1):
            # input feeding: concate y_tm1 and previous attentional vector
            x = torch.cat([y_tm1_embed.squeeze(0), att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)  # E.q. (6)
            scores.append(score_t)

            att_tm1 = att_t
            hidden = h_t, cell_t

        scores = torch.stack(scores)  # T-delta-1, batch_size, voc_size
        word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_sents_var[1:-delta].view(-1))
        ce_loss = word_loss / batch_size

        eos = self.vocab.tgt['</s>']
        if args.cuda:
            sample_ends = torch.cuda.ByteTensor([0] * batch_size)
            all_ones = torch.cuda.ByteTensor([1] * batch_size)
        else:
            sample_ends = torch.ByteTensor([0] * batch_size)
            all_ones = torch.ByteTensor([1] * batch_size)
        offset = torch.mul(torch.range(0, batch_size - 1), len(self.vocab.tgt)).long()

        # p_t = F.softmax(score_t)
        # y_t = torch.multinomial(p_t, num_samples=1).squeeze(1)
        # y_t = y_t.detach()
        # use ground truth as the start token of RL to avoid the first sample is <eos>
        samples = [tgt_sents_var[-delta].view(-1)]
        baselines = []
        sample_losses = []
        epsilon = Variable(torch.ones(batch_size * len(self.vocab.tgt)) * 1e-20, requires_grad=False)

        if args.cuda:
            epsilon = epsilon.cuda()
            offset = offset.cuda()

        t = tgt_word_embed.size(0) - 1
        while t < args.decode_max_time_step:
            t += 1

            # (sample_size)
            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)  # E.q. (6)
            p_t = F.softmax(score_t)

            detach_h_t = h_t.detach()
            baselines.append(self.baseline(detach_h_t).squeeze(1))

            y_t = torch.multinomial(p_t, num_samples=1).squeeze(1)
            y_t = y_t.detach()

            y_t_offset = y_t.data + offset

            samples.append(y_t)
            p_t = p_t.view(-1)
            p_t = -torch.log(p_t + epsilon)
            sample_losses.append(p_t[y_t_offset])

            sample_ends |= torch.eq(y_t, eos).byte().data
            if torch.equal(sample_ends, all_ones):
                break

            att_tm1 = att_t
            hidden = h_t, cell_t

        # transpose tgt_sents_var to batch_size, max_len
        # t_tgt_sents_var = torch.transpose(tgt_sents_var[:-(delta + 1)], 0, 1)
        # print("batch_size and len of CE: ", t_tgt_sents_var.size())

        incomplete_ground_truth = [t.view(-1) for t in tgt_sents_var[:-delta]]
        # for ss in incomplete_ground_truth:
        #     print(ss.size())
        '''The following line is because: the T-s steps might have already included the <eos> token,
         in this case no RL loss follows.'''
        samples = incomplete_ground_truth + samples[1:]
        completed_samples = [list() for _ in range(batch_size)]
        rewards = [-1] * batch_size
        mask_samples = [list() for _ in range(len(samples))]
        # print("tot len: ", len(samples))
        for j, y_t in enumerate(samples):
            # iterate over batch_size
            # print(y_t.size())
            for i, sample_word in enumerate(y_t.cpu().data):
                if (len(completed_samples[i]) == 0 or completed_samples[i][-1] != eos) and rewards[i] == -1:
                    completed_samples[i].append(sample_word)
                    mask_samples[j].append(1.0)
                else:
                    mask_samples[j].append(0.0)
                    if rewards[i] == -1:
                        if len(completed_samples[i]) == 2:
                            rewards[i] = 0.0
                        else:
                            rewards[i] = get_reward(tgt_sents_tokens[i][1:-1], word2id(completed_samples[i][1:-1],
                                                    self.vocab.tgt.id2word), reward_type, len(samples)-1)

        # Clean up: if no <eos> is predicted, we still calculate rewards
        for i in range(batch_size):
            if rewards[i] == -1:
                if len(completed_samples[i]) == 1:
                    rewards[i] = 0.0
                else:
                    rewards[i] = get_reward(tgt_sents_tokens[i][1:-1],
                                               word2id(completed_samples[i][1:],
                                                       self.vocab.tgt.id2word), reward_type, len(samples)-1)

        if not 'delta' in reward_type:
            rewards = Variable(torch.FloatTensor(rewards), requires_grad=False)
            if args.cuda:
                rewards = rewards.cuda()
        else:
            new_rewards = []
            # starts from 0, because in line-519 we use the ground truth as the first token of RL and remove duplicates
            for i in range(0, len(samples)):
                tmp = []
                for j in range(batch_size):
                    if type(rewards[j]) is float:
                        tmp.append(rewards[j])
                    else:
                        tmp.append(rewards[j][i])
                var = Variable(torch.FloatTensor(tmp), requires_grad=False)
                if args.cuda:
                    var = var.cuda()
                new_rewards.append(var)
            rewards = new_rewards
        mask_samples = Variable(torch.FloatTensor(mask_samples), requires_grad=False)

        loss_b = []
        loss_t = []
        if args.cuda:
            mask_samples = mask_samples.cuda()
        # print("mask_samples size: ", mask_samples.size())
        len_CE = len(incomplete_ground_truth)
        # print("len_ce: ", len_CE)
        mask_sum = 0.0

        for i in range(len_CE, len(samples)):
            sample_ind = i - len_CE
            b_t = baselines[sample_ind]
            mask_t = mask_samples[i]  # Caution here!
            prob_t = sample_losses[sample_ind]

            b_t_detach = b_t.detach()
            if not 'delta' in reward_type:
                prob_t = (rewards - b_t_detach) * prob_t
                b_t = (rewards - b_t).pow(2)
            else:
                prob_t = (rewards[i] - b_t_detach) * prob_t
                b_t = (rewards[i] - b_t).pow(2)

            mask_sum += sum(mask_t.data)
            if 0.0 in mask_t.data:
                # print("mask and prob size: ", mask_t.size(), prob_t.size())
                prob_t = mask_t * prob_t
                b_t = mask_t * b_t

            loss_b.append(b_t)
            loss_t.append(prob_t)

        # print("one iter!")
            # print("rewards: ", rewards)

        loss_b = torch.cat(loss_b, 0)  # max_len, batch_size
        prob_t = torch.cat(loss_t, 0)
        # print("before sum loss b: ", loss_b)
        # print("before sum prob: ", prob_t)
        # print(len(loss_b), mask_sum)
        sum_loss_b = torch.sum(loss_b) / mask_sum
        sum_prob_t = torch.sum(prob_t) / batch_size

        if to_word:
            for i, src_sent_samples in enumerate(completed_samples):
                print("Ground truth: ", ' '.join(tgt_sents_tokens[i]))
                completed_samples[i] = word2id(src_sent_samples, self.vocab.tgt.id2word)
                print(' '.join(completed_samples[i]))
        if not 'delta' in reward_type:
            mean_rewards = torch.mean(rewards)
        else:
            mean_rewards = np.array([torch.mean(r) for r in rewards]).mean()

        return ce_loss, sum_loss_b, sum_prob_t, mean_rewards

    def sample_with_loss(self, src_sents, tgt_sents, sample_size=None, to_word=False, reward_type="bleu"):
        if not type(src_sents[0]) == list:
            src_sents = [src_sents]
        if not sample_size:
            sample_size = args.sample_size

        src_sents_num = len(src_sents)
        batch_size = src_sents_num * sample_size
        src_sents_var = to_input_variable(src_sents, self.vocab.src, cuda=args.cuda, is_test=False)
        src_encoding, (dec_init_state, dec_init_cell) = self.encode(src_sents_var, [len(s) for s in src_sents])

        src_encoding = src_encoding.repeat(1, sample_size, 1)
        dec_init_state = dec_init_state.repeat(sample_size, 1)
        dec_init_cell = dec_init_cell.repeat(sample_size, 1)

        src_encoding_att_linear = tensor_transform(self.att_src_linear, src_encoding)
        src_encoding = src_encoding.t()
        src_encoding_att_linear = src_encoding_att_linear.t()

        hidden = (dec_init_state, dec_init_cell)
        new_tensor = dec_init_state.data.new

        att_tm1 = Variable(new_tensor(batch_size, self.args.hidden_size).zero_())
        y_0 = Variable(torch.LongTensor([self.vocab.tgt['<s>'] for _ in xrange(batch_size)]))
        epsilon = Variable(torch.ones(batch_size*len(self.vocab.tgt)) * 1e-20, requires_grad=False)

        eos = self.vocab.tgt['</s>']
        if args.cuda:
            sample_ends = torch.cuda.ByteTensor([0] * batch_size)
            all_ones = torch.cuda.ByteTensor([1] * batch_size)
        else:
            sample_ends = torch.ByteTensor([0] * batch_size)
            all_ones = torch.ByteTensor([1] * batch_size)

        # eos_batch = torch.LongTensor([eos] * batch_size)
        offset = torch.mul(torch.range(0, batch_size - 1), len(self.vocab.tgt)).long()
        if args.cuda:
            y_0 = y_0.cuda()
            epsilon = epsilon.cuda()
            # eos_batch = eos_batch.cuda()
            offset = offset.cuda()
            # sample_ends.cuda()
            # all_ones.cuda()
            
        #    print("offset: ", type(offset))
         #   print("sample_ends: ", type(sample_ends))
         #   print("ones:", type(all_ones))
        samples = [y_0]

        baselines = []
        sample_losses = []
        t = 0
        while t < args.decode_max_time_step:
            t += 1

            # (sample_size)
            y_tm1 = samples[-1]

            y_tm1_embed = self.tgt_embed(y_tm1)

            x = torch.cat([y_tm1_embed, att_tm1], 1)

            # h_t: (batch_size, hidden_size)
            h_t, cell_t = self.decoder_lstm(x, hidden)
            h_t = self.dropout(h_t)

            ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encoding, src_encoding_att_linear)

            att_t = F.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
            att_t = self.dropout(att_t)

            score_t = self.readout(att_t)  # E.q. (6)
            p_t = F.softmax(score_t)

            detach_h_t = h_t.detach()
            baselines.append(self.baseline(detach_h_t).squeeze(1))

            y_t = torch.multinomial(p_t, num_samples=1).squeeze(1)
            y_t = y_t.detach()

            y_t_offset = y_t.data + offset

            samples.append(y_t)
            p_t = p_t.view(-1)
            p_t = -torch.log(p_t + epsilon)
            sample_losses.append(p_t[y_t_offset])
            
            sample_ends |= torch.eq(y_t, eos).byte().data
            if torch.equal(sample_ends, all_ones):
                break
            # if torch.equal(y_t.data, eos_batch):
            #     break

            att_tm1 = att_t
            hidden = h_t, cell_t

        # post-processing
        completed_samples = [list([list() for _ in xrange(sample_size)]) for _ in xrange(src_sents_num)]
        mask_sample = []
        rewards = [-1] * batch_size
        for j, y_t in enumerate(samples):
            for i, sampled_word in enumerate(y_t.cpu().data):
                src_sent_id = i % src_sents_num
                sample_id = i / src_sents_num

                if (len(completed_samples[src_sent_id][sample_id]) == 0 or completed_samples[src_sent_id][sample_id][-1] != eos) and rewards[sample_id] == -1:
                    completed_samples[src_sent_id][sample_id].append(sampled_word)
                    mask_sample.append(1.0)
                else:
                    mask_sample.append(0.0)
                    if rewards[i] == -1:
                        if len(completed_samples[src_sent_id][sample_id]) == 2:
                            rewards[i] = 0.0
                        else:
                            rewards[i] = get_reward(tgt_sents[src_sent_id][1:-1],
                                                word2id(completed_samples[src_sent_id][sample_id][1:-1],
                                                        self.vocab.tgt.id2word), reward_type,len(samples))
        # if no <eos> is predicted, we still calculate rewards
        for i in range(batch_size):
            src_sent_id = i % src_sents_num
            sample_id = i / src_sents_num
            if rewards[i] == -1:
                if len(completed_samples[src_sent_id][sample_id]) == 1:
                    rewards[i] = 0.0
                else:
                    rewards[i] = get_reward(tgt_sents[src_sent_id][1:-1],
                                               word2id(completed_samples[src_sent_id][sample_id][1:],
                                                       self.vocab.tgt.id2word), reward_type, len(samples))
        # for e in rewards:
        #     print(e)
        #     print(len(samples)-1)
        #     print(len(e))
        if not 'delta' in reward_type:
            rewards = Variable(torch.FloatTensor(rewards), requires_grad=False)
            if args.cuda:
                rewards = rewards.cuda()
        else:
            new_rewards = []
            # FIXME: change here, because delta rewards should be received from the first sampling token, which should match with b_t and p_t
            for i in range(1, len(samples)):
                tmp = []
                for j in range(batch_size):
                    if type(rewards[j]) is float:
                        tmp.append(rewards[j])
                    else:
                        tmp.append(rewards[j][i])
                var = Variable(torch.FloatTensor(tmp), requires_grad=False)
                if args.cuda:
                    var = var.cuda()
                new_rewards.append(var)
            rewards = new_rewards

        mask_sample = Variable(torch.FloatTensor(mask_sample), requires_grad=False)

        # len(baselines) = len(mask_sample)/batch_size - 1
        # print("Check baseline len and mask len: ", len(baselines), len(mask_sample)/batch_size)
        loss_b = []
        loss_t = []
        if args.cuda:
            mask_sample = mask_sample.cuda()
        # neg_log_probs = Variable(torch.zeros(batch_size), requires_grad=False)
        for i in range(len(samples)-1):
            b_t = baselines[i]
            # FIXme: the correct index of mask_t
            mask_t = mask_sample[(i+1)*batch_size:(i+2)*batch_size]
            prob_t = sample_losses[i]

            b_t_detach = b_t.detach()
            if not 'delta' in reward_type:
                prob_t = (rewards - b_t_detach) * prob_t
                b_t = (rewards - b_t).pow(2)
            else:
                prob_t = (rewards[i] - b_t_detach) * prob_t
                b_t = (rewards[i] - b_t).pow(2)

            if 0.0 in mask_t.data:
                prob_t = mask_t * prob_t
                b_t = mask_t * b_t

            loss_b.append(b_t)
            loss_t.append(prob_t)

        # print("rewards: ", rewards)
        mask_sum = sum(mask_sample)
        loss_b = torch.cat(loss_b, 0) # max_len, batch_size
        prob_t = torch.cat(loss_t, 0)
        # print("before sum loss b: ", loss_b)
        # print("before sum prob: ", prob_t)
        sum_loss_b = torch.sum(loss_b)/mask_sum
        sum_prob_t = torch.sum(prob_t)/batch_size

        # print("sum of mask: ", mask_sum.data)
        # print("sum_prob_t: ", sum_prob_t.data)
        # print("sum of b t: ", sum_loss_b.data)
        if to_word:
            for i, src_sent_samples in enumerate(completed_samples):
                tgt_sent_id = i % src_sents_num
                print("Ground truth: ", ' '.join(tgt_sents[tgt_sent_id]))
                completed_samples[i] = word2id(src_sent_samples, self.vocab.tgt.id2word)
                for e in completed_samples[i]:
                    print(' '.join(e))
        # print("mean reward: ", torch.mean(rewards))
        if not 'delta' in reward_type:
            mean_rewards = torch.mean(rewards)
        else:
            mean_rewards = np.array([torch.mean(r) for r in rewards]).mean()
        return sum_loss_b, sum_prob_t, mean_rewards

    def attention(self, h_t, src_encoding, src_linear_for_att):
        # (1, batch_size, attention_size) + (src_sent_len, batch_size, attention_size) =>
        # (src_sent_len, batch_size, attention_size)
        att_hidden = F.tanh(self.att_h_linear(h_t).unsqueeze(0).expand_as(src_linear_for_att) + src_linear_for_att)

        # (batch_size, src_sent_len)
        att_weights = F.softmax(tensor_transform(self.att_vec_linear, att_hidden).squeeze(2).permute(1, 0))

        # (batch_size, hidden_size * 2)
        ctx_vec = torch.bmm(src_encoding.permute(1, 2, 0), att_weights.unsqueeze(2)).squeeze(2)

        return ctx_vec, att_weights

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask:
            att_weight.data.masked_fill_(mask, -float('inf'))
        att_weight = F.softmax(att_weight)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def save(self, path):
        print('save parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(params, path)


def to_input_variable(sents, vocab, cuda=False, is_test=False):
    """
    return a tensor of shape (src_sent_len, batch_size)
    """

    word_ids = word2id(sents, vocab)
    sents_t, masks = input_transpose(word_ids, vocab['<pad>'])

    sents_var = Variable(torch.LongTensor(sents_t), volatile=is_test, requires_grad=False)
    if cuda:
        sents_var = sents_var.cuda()

    return sents_var


def evaluate_loss(model, data, crit):
    model.eval()
    cum_loss = 0.
    cum_tgt_words = 0.
    for src_sents, tgt_sents in data_iter(data, batch_size=args.batch_size, shuffle=False):
        pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`
        src_sents_len = [len(s) for s in src_sents]

        src_sents_var = to_input_variable(src_sents, model.vocab.src, cuda=args.cuda, is_test=True)
        tgt_sents_var = to_input_variable(tgt_sents, model.vocab.tgt, cuda=args.cuda, is_test=True)

        # (tgt_sent_len, batch_size, tgt_vocab_size)
        scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
        loss = crit(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))

        cum_loss += loss.data[0]
        cum_tgt_words += pred_tgt_word_num

    loss = cum_loss / cum_tgt_words
    return loss


def init_training(args):
    vocab = torch.load(args.vocab)

    if args.model_type == "rl" or args.model_type == "mixer":
        assert args.load_model is not None, "Path to pretrained model is not provided!!"
        print('load model from [%s] for RL training' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(args, vocab)
        model.load_state_dict(state_dict)
    else:
        model = NMT(args, vocab)
    model.train()

    if args.uniform_init:
        print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-args.uniform_init, args.uniform_init)

    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0
    nll_loss = nn.NLLLoss(weight=vocab_mask, size_average=False)
    cross_entropy_loss = nn.CrossEntropyLoss(weight=vocab_mask, size_average=False)

    if args.cuda:
        model = model.cuda()
        nll_loss = nll_loss.cuda()
        cross_entropy_loss = cross_entropy_loss.cuda()

    if args.model_type == "ml":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        # rl
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.)

    return vocab, model, optimizer, nll_loss, cross_entropy_loss


def train(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')

    dev_data_src = read_corpus(args.dev_src, source='src')
    dev_data_tgt = read_corpus(args.dev_tgt, source='tgt')

    train_data = zip(train_data_src, train_data_tgt)
    dev_data = zip(dev_data_src, dev_data_tgt)

    vocab, model, optimizer, nll_loss, cross_entropy_loss = init_training(args)

    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = cum_batches = report_examples = epoch = valid_num = best_model_iter = 0
    delta = args.delta_steps
    hist_valid_scores = []
    train_time = begin_time = time.time()
    reward_val = 0.
    if args.model_type=='ml':
        print('Begin Maximum Likelihood training..')
    elif args.model_type=='rl':
        print('Begin RL-based training..')
    total_loss_baseline = 0.
    optimizer.zero_grad()
    while True:
        epoch += 1
        for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
            train_iter += 1

            # sent_len, voc_size
            src_sents_var = to_input_variable(src_sents, vocab.src, cuda=args.cuda)
            tgt_sents_var = to_input_variable(tgt_sents, vocab.tgt, cuda=args.cuda)

            batch_size = len(src_sents)
            src_sents_len = [len(s) for s in src_sents]
            tgt_sents_len = [len(s) for s in tgt_sents]
            pred_tgt_word_num = sum(len(s[1:]) for s in tgt_sents) # omitting leading `<s>`

            if args.model_type == 'ml':
            # (tgt_sent_len, batch_size, tgt_vocab_size)
                scores = model(src_sents_var, src_sents_len, tgt_sents_var[:-1])
                word_loss = cross_entropy_loss(scores.view(-1, scores.size(2)), tgt_sents_var[1:].view(-1))
                loss = word_loss / batch_size
                word_loss_val = word_loss.data[0]
                loss_val = loss.data[0]
                report_loss += word_loss_val
                cum_loss += word_loss_val

            elif args.model_type == 'rl':
                loss_b, loss_rl, avg_reward = model.sample_with_loss(src_sents, tgt_sents, args.sample_size, False, reward_type=args.reward_type)
                loss = loss_b + loss_rl
                loss_val = loss.data[0]
                reward_val = avg_reward.data[0]
                report_loss += loss_val * batch_size
                cum_loss += loss_val * batch_size
                total_loss_baseline += loss_b.data[0]*batch_size

            elif args.model_type == 'mixer':
                if epoch % args.XER == 0:
                    delta += delta
                max_len = max(tgt_sents_len)
                if delta >= max_len - 1:
                    # Totally switch to RL training
                    loss_b, loss_rl, avg_reward = model.sample_with_loss(src_sents, tgt_sents, args.sample_size, False,
                                                                         reward_type=args.reward_type)
                    loss = loss_b + loss_rl
                else:
                    src_encodings, init_ctx_vec = model.encode(src_sents_var, src_sents_len)
                    ce_loss, loss_b, loss_rl, avg_reward = model.mixer(src_encodings, init_ctx_vec, tgt_sents_var,
                                                                            delta, tgt_sents, cross_entropy_loss, args.reward_type, False)
                    loss = ce_loss + loss_b + loss_rl
                loss_val = loss.data[0]
                reward_val = avg_reward.data[0]
                report_loss += loss_val * batch_size
                cum_loss += loss_val * batch_size
                total_loss_baseline += loss_b.data[0] * batch_size

            loss.backward()
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_grad)
            if train_iter % args.update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()


            report_tgt_words += pred_tgt_word_num
            cum_tgt_words += pred_tgt_word_num
            report_examples += batch_size
            cum_examples += batch_size
            cum_batches += batch_size

            if train_iter % args.log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. reward %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         reward_val,
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                print('baseline loss %.3f'%(total_loss_baseline/report_examples))
                train_time = time.time()
                total_loss_baseline = report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % args.valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_batches,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_batches = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)
                model.eval()

                dev_loss = 0.0
                dev_ppl = 0.0

                # compute dev. ppl and bleu
                if args.model_type == 'ml':
                    dev_loss = evaluate_loss(model, dev_data, cross_entropy_loss)
                    dev_ppl = np.exp(dev_loss)

                if args.valid_metric in ['bleu', 'word_acc', 'sent_acc']:
                    dev_hyps = decode(model, dev_data)
                    dev_hyps = [hyps[0] for hyps in dev_hyps]
                    if args.valid_metric == 'bleu':
                        valid_metric = get_bleu([tgt for src, tgt in dev_data], dev_hyps)
                    else:
                        valid_metric = get_acc([tgt for src, tgt in dev_data], dev_hyps, acc_type=args.valid_metric)
                    print('validation: iter %d, dev. ppl %f, dev. %s %f' % (train_iter, dev_ppl, args.valid_metric, valid_metric),
                          file=sys.stderr)
                else:

                    valid_metric = -dev_ppl
                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl),
                          file=sys.stderr)

                model.train()

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                is_better_than_last = len(hist_valid_scores) == 0 or valid_metric > hist_valid_scores[-1]
                hist_valid_scores.append(valid_metric)

                if valid_num > args.save_model_after:
                    model_file = args.save_to + '.iter%d.bin' % train_iter
                    print('save model to [%s]' % model_file, file=sys.stderr)
                    model.save(model_file)

                if (not is_better_than_last) and args.lr_decay:
                    lr = optimizer.param_groups[0]['lr'] * args.lr_decay
                    print('decay learning rate to %f' % lr, file=sys.stderr)
                    optimizer.param_groups[0]['lr'] = lr

                if is_better:
                    patience = 0
                    best_model_iter = train_iter

                    if valid_num > args.save_model_after:
                        print('save currently the best model ..', file=sys.stderr)
                        model_file_abs_path = os.path.abspath(model_file)
                        symlin_file_abs_path = os.path.abspath(args.save_to + '.bin')
                        os.system('ln -sf %s %s' % (model_file_abs_path, symlin_file_abs_path))
                else:
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    if epoch <= args.run_with_no_patience:
                        patience = 0
                    if patience >= args.patience:
                        print('early stop!', file=sys.stderr)
                        print('the best model is from iteration [%d]' % best_model_iter, file=sys.stderr)
                        exit(0)


def get_bleu(references, hypotheses):
    # compute BLEU
    bleu_score = corpus_bleu([[ref[1:-1]] for ref in references],
                             [hyp[1:-1] for hyp in hypotheses])

    return bleu_score


def get_acc(references, hypotheses, acc_type='word'):
    assert acc_type == 'word_acc' or acc_type == 'sent_acc' or acc_type == 'f1'
    cum_acc = 0.

    for ref, hyp in zip(references, hypotheses):
        ref = ref[1:-1]
        hyp = hyp[1:-1]
        if acc_type == 'word_acc':
            acc = len([1 for ref_w, hyp_w in zip(ref, hyp) if ref_w == hyp_w]) / float(len(hyp) + 1e-6)
        elif acc_type == 'sent_acc':
            acc = 1. if all(ref_w == hyp_w for ref_w, hyp_w in zip(ref, hyp)) else 0.
        elif acc_type == 'f1':
            acc = calc_f1(ref, hyp)
        cum_acc += acc

    acc = cum_acc / len(hypotheses)
    return acc


def decode(model, data, verbose=True):
    """
    decode the dataset and compute sentence level acc. and BLEU.
    """
    hypotheses = []
    begin_time = time.time()

    if type(data[0]) is tuple:
        for src_sent, tgt_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

            if verbose:
                print('*' * 50)
                print('Source: ', ' '.join(src_sent))
                print('Target: ', ' '.join(tgt_sent))
                print('Top Hypothesis: ', ' '.join(hyps[0]))
    else:
        for src_sent in data:
            hyps = model.translate(src_sent)
            hypotheses.append(hyps)

            if verbose:
                print('*' * 50)
                print('Source: ', ' '.join(src_sent))
                print('Top Hypothesis: ', ' '.join(hyps[0]))

    elapsed = time.time() - begin_time

    print('decoded %d examples, took %d s' % (len(data), elapsed), file=sys.stderr)

    return hypotheses


def test(args):
    test_data_src = read_corpus(args.test_src, source='src')
    test_data_tgt = read_corpus(args.test_tgt, source='tgt')
    test_data = zip(test_data_src, test_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        state_dict = params['state_dict']

        model = NMT(saved_args, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    hypotheses = decode(model, test_data,False)
    top_hypotheses = [hyps[0] for hyps in hypotheses]

    bleu_score = get_bleu([tgt for src, tgt in test_data], top_hypotheses)
    word_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'word_acc')
    sent_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'sent_acc')
    f1_acc = get_acc([tgt for src, tgt in test_data], top_hypotheses, 'f1')
    print('Corpus Level BLEU: %f, word level acc: %f, sentence level acc: %f, f1 acc: %f' % (bleu_score, word_acc, sent_acc, f1_acc),
          file=sys.stderr)

    if args.save_to_file:
        print('save decoding results to %s' % args.save_to_file, file=sys.stderr)
        with open(args.save_to_file, 'w') as f:
            for hyps in hypotheses:
                f.write(' '.join(hyps[0][1:-1]) + '\n')

        if args.save_nbest:
            nbest_file = args.save_to_file + '.nbest'
            print('save nbest decoding results to %s' % nbest_file, file=sys.stderr)
            with open(nbest_file, 'w') as f:
                for src_sent, tgt_sent, hyps in zip(test_data_src, test_data_tgt, hypotheses):
                    print('Source: %s' % ' '.join(src_sent), file=f)
                    print('Target: %s' % ' '.join(tgt_sent), file=f)
                    print('Hypotheses:', file=f)
                    for i, hyp in enumerate(hyps, 1):
                        print('[%d] %s' % (i, ' '.join(hyp)), file=f)
                    print('*' * 30, file=f)


def sample(args):
    train_data_src = read_corpus(args.train_src, source='src')
    train_data_tgt = read_corpus(args.train_tgt, source='tgt')
    train_data = zip(train_data_src, train_data_tgt)

    if args.load_model:
        print('load model from [%s]' % args.load_model, file=sys.stderr)
        params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        opt = params['args']
        state_dict = params['state_dict']

        model = NMT(opt, vocab)
        model.load_state_dict(state_dict)
    else:
        vocab = torch.load(args.vocab)
        model = NMT(args, vocab)

    model.eval()

    if args.cuda:
        model = model.cuda()

    print('begin sampling')

    check_every = 10
    train_iter = cum_samples = 0
    train_time = time.time()
    for src_sents, tgt_sents in data_iter(train_data, batch_size=args.batch_size):
        train_iter += 1
        samples = model.sample(src_sents, sample_size=args.sample_size, to_word=True)
        cum_samples += sum(len(sample) for sample in samples)

        if train_iter % check_every == 0:
            elapsed = time.time() - train_time
            print('sampling speed: %d/s' % (cum_samples / elapsed), file=sys.stderr)
            cum_samples = 0
            train_time = time.time()

        for i, tgt_sent in enumerate(tgt_sents):
            print('*' * 80)
            print('target:' + ' '.join(tgt_sent))
            tgt_samples = samples[i]
            print('samples:')
            for sid, sample in enumerate(tgt_samples, 1):
                print('[%d] %s' % (sid, ' '.join(sample[1:-1])))
            print('*' * 80)


if __name__ == '__main__':
    args = init_config()
    print(args, file=sys.stderr)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'sample':
        sample(args)
    elif args.mode == 'test':
        test(args)
    else:
        raise RuntimeError('unknown mode')
