from torch.autograd import Variable
from torch.nn.functional import softmax
import pickle
import argparse

import torch
from tqdm import tqdm
import time
from datetime import timedelta

PAD, CLS = '[PAD]', '[CLS]'


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def masked_softmax(vector, seq_lens):
    mask = vector.new(vector.size()).zero_()
    for i in range(seq_lens.size(0)):
        mask[i, :, :seq_lens[i]] = 1
    mask = Variable(mask, requires_grad=False)
    # mask = None
    if mask is None:
        result = softmax(vector, dim=-1)
    else:
        result = softmax(vector * mask, dim=-1)
        result = result * mask
        result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def seperate_seq(sequence_output, doc_len, ques_len, option_len):
    doc_seq_output = sequence_output.new(sequence_output.size()).zero_()
    doc_ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_seq_output = sequence_output.new(sequence_output.size()).zero_()
    ques_option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    option_seq_output = sequence_output.new(sequence_output.size()).zero_()
    for i in range(doc_len.size(0)):
        doc_seq_output[i, :doc_len[i]] = sequence_output[i, 1:doc_len[i]+1]
        doc_ques_seq_output[i, :doc_len[i] + ques_len[i]] = sequence_output[i, :doc_len[i] + ques_len[i]]
        ques_seq_output[i, :ques_len[i]] = sequence_output[i, doc_len[i] + 2:doc_len[i] + ques_len[i] + 2]
        ques_option_seq_output[i, :ques_len[i]+option_len[i]] = sequence_output[i, doc_len[i] + 1:doc_len[i] + ques_len[i] + option_len[i] + 1]
        # print(option_len[i],len(sequence_output[i]),doc_len[i] , ques_len[i])
        option_seq_output[i, :option_len[i]] = sequence_output[i,
                                                 doc_len[i] + ques_len[i] + 2:doc_len[i] + ques_len[i] + option_len[
                                                   i] + 2]
        
    return doc_ques_seq_output, ques_option_seq_output, doc_seq_output, ques_seq_output, option_seq_output


def parse_mc(input_file, answer_file, max_pad_length, dg):
    if 'dev' in input_file:
        _sentences = dg.test_dcmn_srcs
        _labels = dg.test_dcmn_labels
    else:
        _sentences = dg.train_dcmn_srcs
        _labels = dg.train_dcmn_labels

    sentences = _sentences
    labels = _labels

    q_id = [i+1 for i in range(len(labels))]
    article = [u[0] for u in sentences]
    question = [u[1] for u in sentences]
    cts = []
    for i in range(max_pad_length-2):
        cts.append([u[i+2] for u in sentences])
    y = labels

    return article, question, cts, y, q_id


def remove_unk(seq_srcs, dcmn_outs, key_choices):
    p = 0
    k = 0
    seq_srcs_ok = []
    for out in dcmn_outs:
        while p < len(seq_srcs) and seq_srcs[p].find('[UNK]') == -1:
            seq_srcs_ok.append(seq_srcs[p])
            p += 1
            k = 0
        if p < len(seq_srcs):
            seq_srcs[p] = seq_srcs[p].replace('[UNK]', '[MASK] ' + key_choices[p][k][out] + ' [MASK]', 1)
            k += 1
    while p < len(seq_srcs) and seq_srcs[p].find('[UNK]') == -1:
        seq_srcs_ok.append(seq_srcs[p])
        p += 1
    return seq_srcs_ok


def decode_sentence(symbols, config):
    sentences = []
    for symbol_sen in symbols:
        words = config.tokenizer.convert_ids_to_tokens(symbol_sen)
        temp = ''
        for word in words:
            if word == '[SEP]':
                break
            if word.startswith('##'):
                word = word[2:]
                temp += word
            else:
                temp += ' '
                temp += word
        sentences.append(temp)
    return sentences


class DatasetIterater(object):
    def __init__(self, seq_dataset, dcmn_dataset, batch_size):
        self.batch_size = batch_size
        self.seq_dataset = seq_dataset
        self.dcmn_dataset = dcmn_dataset
        self.n_batches = len(seq_dataset) // batch_size
        self.residue = False
        if len(seq_dataset) % self.n_batches != 0:
            self.residue = True
        self.index = 0

        indices = []
        for i, u in enumerate(self.seq_dataset):
            seq_src = u[0]
            if len(indices) > 0:
                indices.append(indices[i-1]+seq_src.count('[UNK]'))
            else:
                indices.append(seq_src.count('[UNK]'))
        indices.append(0)
        self.indices = indices

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            p = self.index * self.batch_size
            q = len(self.seq_dataset)
            seq_batches = self.seq_dataset[p: q]
            dcmn_batches = self.dcmn_dataset[self.indices[p-1]: self.indices[q-1]]
            self.index += 1
            return seq_batches, dcmn_batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            p = self.index * self.batch_size
            q = (self.index + 1) * self.batch_size
            seq_batches = self.seq_dataset[p: q]
            dcmn_batches = self.dcmn_dataset[self.indices[p-1]: self.indices[q-1]]
            self.index += 1
            return seq_batches, dcmn_batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(seq_dataset, dcmn_dataset, config):
    iter = DatasetIterater(seq_dataset, dcmn_dataset, config.batch_size)
    return iter

def build_eval_iterator(seq_dataset, dcmn_dataset, config):
    iter = DatasetIterater(seq_dataset, dcmn_dataset, config.eval_batch_size)
    return iter




