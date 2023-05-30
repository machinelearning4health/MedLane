import pickle
import re
import torch
from keras.preprocessing.sequence import pad_sequences
from preprocess import SwagExample, convert_examples_to_features
from utils import select_field
import os


def word_tokenize(sentence, tokenizer):
    tokens = tokenizer.tokenize(sentence)
    temp = ''
    for word in tokens:
        if word.startswith('##'):
            word = word[2:]
            temp += word
        else:
            temp += ' '
            temp += word
    temp = temp.strip()

    return temp.split(' ')


def is_similar(word_1, word_2, stop=False):
    word_1 = simplify(word_1)
    word_2 = simplify(word_2)
    flag = False
    if ('(' in word_1 or '(' in word_2) and not stop:
        _word_1 = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", word_1)
        _word_2 = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", word_2)
        flag = is_similar(_word_1, _word_2, True)

    return word_1 == word_2 or \
           word_1 in word_2 or \
           word_2 in word_1 or \
           min_distance(word_1, word_2) <= 2 or \
           flag


def simplify(word):
    new_word = word.lower().replace('( ', '(').replace(' )', ')').replace('  ', ' ').replace(' ,', ',').strip()
    return new_word


def min_distance(str1, str2):
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


def get_train_src_tar_txt(train_txt_path):
    src = []
    tar_1 = []
    tar_2 = []
    txt = ''
    try:
        txt += open(train_txt_path, 'r').read()
    except:
        txt += open(train_txt_path, 'r', encoding='utf-8').read()

    txt = txt.split('\n\n')
    for para in txt:
        sentences = para.split('\n')
        if len(sentences) < 2:
            continue
        for sid, sentence in enumerate(sentences[0:3]):
            if sid == 0:
                src.append(sentence)
            elif sid == 1:
                tar_1.append(sentence)
            elif sid == 2:
                tar_2.append(sentence)
    return src, tar_1, tar_2


def get_test_src_tar_txt(test_txt_path, tokenizer):
    txt = open(test_txt_path, 'r').read()
    #     txt = txt.lower()
    txt = txt.split('\n\n')
    src = []
    tar_1 = []
    tar_2 = []
    cudics = []
    tars = []
    for para in txt:
        sentences = para.split('\n')
        src_sentence = ''
        _tars = []
        _cudics = []
        if len(sentences) < 2 or len(sentences[0]) < 3 or len(sentences[1]) < 3:
            continue
        for sid, sentence in enumerate(sentences):
            if sid == 0:
                src.append(sentence)
            else:
                cudic = {}
                sentence = sentence[2:].lower()
                sentence = sentence.replace('].', '] .')
                text = re.sub('\[[^\[\]]*\]', '', sentence)
                pairs = re.findall('[^\[\] ]+\[[^\[\]]+\]', sentence)
                for pair in pairs:
                    pair = re.split('[\[\]]', pair)
                    cudic[pair[0]] = pair[1]
                words = word_tokenize(text, tokenizer)
                for wid, word in enumerate(words):
                    if word in cudic.keys():
                        words[wid] = cudic[word]
                new_text = ' '.join(words)
                if sid == 1:
                    tar_1.append(new_text)
                elif sid == 2:
                    tar_2.append(new_text)
                _tars.append(new_text)
                _cudics.append(cudic)
        tars.append(_tars)
        cudics.append(_cudics)

    with open('./data/test_cudics.pkl', 'wb') as f:
        pickle.dump(cudics, f)
    with open('./data/test_tars.pkl', 'wb') as f:
        pickle.dump(tars, f)

    return src, tar_1, tar_2


def get_one_sample(src_words, key, key_tar, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    if key in abbrs.keys():
        pass
    elif key.upper() in abbrs.keys():
        key = key.upper()
    elif key.lower() in abbrs.keys():
        key = key.lower()

    choices = []

    if key in abbrs.keys() and key_tar is not None:
        temp = [' '.join(src_words), 'what is {} ?'.format(key)]
        label = -1
        skip_cnt = 0
        for index, u in enumerate(abbrs[key]):
            if index - skip_cnt >= max_pad_length - 2:
                break
            if len(u.split(' ')) > 10:
                skip_cnt += 1
                continue
            temp.append(u)
            choices.append(u)
            if is_similar(u, key_tar):
                label = index - skip_cnt
        while len(temp) < max_pad_length:
            temp.append('[PAD]')
            choices.append('[PAD]')

        if len(tokenizer.tokenize(temp[0])) + len(tokenizer.tokenize(temp[1])) + len(
                tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length \
                or label < 0 or label >= max_pad_length - 2:
            return None, None, None
        else:
            return temp, label, choices
    else:
        # return None, None, None
        temp = [' '.join(src_words), 'what is {} ?'.format(key), key]
        choices.append(key)
        while len(temp) < max_pad_length:
            temp.append('[PAD]')
            choices.append('[PAD]')
        if len(tokenizer.tokenize(temp[0])) + len(tokenizer.tokenize(temp[1])) + len(
                tokenizer.tokenize(temp[2])) >= max_dcmn_seq_length:
            return None, None, None
        return temp, 0, choices


def get_dcmn_data_from_gt(src_words, tar_words, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    if tar_words[-1] != '.':
        tar_words.append('.')
    i = 0
    j = 0
    sentences = []
    labels = []
    key_choices = []
    seq_src_words = src_words[:]
    indics = []
    key_ans = {}

    while i < len(src_words):
        if j == len(tar_words):
            break
        if src_words[i] == tar_words[j]:
            i += 1
            j += 1
        else:
            p = i + 1
            q = j + 1

            while p < len(src_words):
                while q < len(tar_words) and tar_words[q] != src_words[p]:
                    q += 1
                if q == len(tar_words):
                    p = p + 1
                    q = j + 1
                else:
                    break
            aft = " ".join(tar_words[j:q])
            for k, word in enumerate(src_words[i:p]):
                temp, label, choices = get_one_sample(src_words, word, aft, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer)
                if temp is not None:
                    sentences.append(temp)
                    labels.append(label)
                    key_choices.append(choices)
                    key_ans[word] = temp[label + 2]
                    seq_src_words[i+k] = '[UNK]'
                    indics.extend([j,q])

            i = p
            j = q


    seq_tar_words = []
    for i,word in enumerate(tar_words):
        if i in indics:
            seq_tar_words.append('[MASK]')
        seq_tar_words.append(word)

    return sentences, labels, ' '.join(seq_src_words), key_ans, key_choices, ' '.join(seq_tar_words)


def get_dcmn_data_from_step1(src_words, masks, k_a, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer):
    sentences = []
    seq_src_words = src_words[:]
    labels = []
    key_choices = []
    for i, mask in enumerate(masks):
        if mask == 0:
            continue
        key = src_words[i]
        if key in abbrs.keys():
            pass
        elif key.upper() in abbrs.keys():
            key = key.upper()
        elif key.lower() in abbrs.keys():
            key = key.lower()

        if key in k_a.keys():
            aft = k_a[key]
        elif key in abbrs.keys() and len(abbrs[key]) == 1:
            aft = abbrs[key][0]
        else:
            aft = None
        temp, label, choices = get_one_sample(src_words, key, aft, abbrs, max_pad_length, max_dcmn_seq_length, tokenizer)
        if temp is not None:
            sentences.append(temp)
            labels.append(label)
            key_choices.append(choices)
            seq_src_words[i] = '[UNK]'

    return sentences, labels, ' '.join(seq_src_words), key_choices


def seq_tokenize(input_data, config):
    ids = []
    for data in input_data:
        words = config.tokenizer.tokenize(data)
        ids.append(words)

    ids = pad_sequences([config.tokenizer.convert_tokens_to_ids(txt) for txt in ids],
                        maxlen=config.max_seq_length, dtype="long", value=0,
                        truncating="post", padding="post")
    masks = [[float(i != 0.0) for i in ii] for ii in ids]

    ids = torch.LongTensor(ids).to(config.seq_device)
    masks = torch.LongTensor(masks).to(config.seq_device)

    return ids, masks


def build_dataset(config):
    abbrs_path = './data/abbrs-all-cased.pkl'
    # txt_path = './data/train(12809).txt'
    txt_path = os.path.join(config.data_dir, config.train_file)
    with open(abbrs_path, 'rb') as f:
        abbrs = pickle.load(f)
    src_txt, tar_1_txt, tar_2_txt = get_train_src_tar_txt(txt_path)
    # src_txt = src_txt[:100]
    # tar_1_txt = tar_1_txt[:100]
    # tar_2_txt = tar_2_txt[:100]

    seq_srcs = []
    seq_tars = []
    dcmn_srcs = []
    dcmn_labels = []
    key_choices = []

    for i, (src, tar) in enumerate(zip(src_txt, tar_1_txt)):
        src = word_tokenize(src, config.tokenizer)
        tar = word_tokenize(tar, config.tokenizer)
        sentences, labels, _src, key_ans, k_c, _tar = get_dcmn_data_from_gt(src, tar, abbrs,
                                                                            max_pad_length=config.num_choices + 2,
                                                                            max_dcmn_seq_length=config.max_seq_length,
                                                                            tokenizer=config.tokenizer)
        if len(sentences) != _src.count('[UNK]'):
            print(i, src, len(sentences))
        dcmn_srcs.extend(sentences)
        dcmn_labels.extend(labels)
        seq_srcs.append(_src)
        seq_tars.append(_tar)
        key_choices.append(k_c)

    for i in range(len(seq_srcs)):
        seq_srcs[i] = '[CLS] ' + seq_srcs[i] + ' [SEP]'

    q_id = [i + 1 for i in range(len(dcmn_labels))]
    article = [u[0] for u in dcmn_srcs]
    question = [u[1] for u in dcmn_srcs]
    cts = []
    for i in range(config.num_choices):
        cts.append([u[i + 2] for u in dcmn_srcs])

    examples = [
        SwagExample(
            swag_id=s5,
            context_sentence=s1,
            start_ending=s2,
            endings=s3,
            label=s4,
        ) for i, (s1, s2, *s3, s4, s5) in
        enumerate(zip(article, question, *cts, dcmn_labels, q_id))
    ]

    features = convert_examples_to_features(examples, config.tokenizer, config.max_seq_length)
    input_ids = select_field(features, 'input_ids')
    input_mask = select_field(features, 'input_mask')
    segment_ids = select_field(features, 'segment_ids')
    doc_len = select_field(features, 'doc_len')
    ques_len = select_field(features, 'ques_len')
    option_len = select_field(features, 'option_len')
    labels = [f.label for f in features]

    dcmn_contents = []
    for i in range(len(input_ids)):
        dcmn_contents.append((input_ids[i], input_mask[i], segment_ids[i], doc_len[i], ques_len[i], option_len[i], labels[i]))

    seq_contents = []
    for i in range(len(seq_srcs)):
        seq_contents.append((seq_srcs[i], seq_tars[i], key_choices[i]))

    return seq_contents, dcmn_contents


def build_dataset_eval(config):
    abbrs_path = './data/abbrs-all-cased.pkl'
    # txt_path = './data/test(2030).txt'
    txt_path = os.path.join(config.data_dir, config.test_file)

    with open(abbrs_path, 'rb') as f:
        abbrs = pickle.load(f)
    src_txt, tar_1_txt, tar_2_txt = get_test_src_tar_txt(txt_path, config.tokenizer)
    seq_srcs = []
    dcmn_srcs = []
    dcmn_labels = []
    key_choices = []

    with open('./data/test_mask_step2_2030.pkl', 'rb') as f:
        mask_step1 = pickle.load(f)

    k_as = []
    for i, (src, tar) in enumerate(zip(src_txt, tar_1_txt)):
        src = word_tokenize(src, config.tokenizer)
        tar = word_tokenize(tar, config.tokenizer)
        sentences, labels, _src, key_ans, _, _tar = get_dcmn_data_from_gt(src, tar, abbrs,
                                                                          max_pad_length=config.num_choices + 2,
                                                                          max_dcmn_seq_length=config.max_seq_length,
                                                                          tokenizer=config.tokenizer)
        k_as.append(key_ans)

    for i, (sts, masks, k_a) in enumerate(zip(src_txt, mask_step1, k_as)):
        sts = word_tokenize(sts, config.tokenizer)
        assert len(sts) == len(masks)
        sentences, labels, _src, k_cs = get_dcmn_data_from_step1(sts, masks, k_a, abbrs,
                                                                        max_pad_length=config.num_choices + 2,
                                                                        max_dcmn_seq_length=config.max_seq_length,
                                                                        tokenizer=config.tokenizer)
        dcmn_srcs.extend(sentences)
        dcmn_labels.extend(labels)
        if len(sentences) != _src.count('[UNK]'):
            print(i, sts)
        seq_srcs.append(_src)
        key_choices.append(k_cs)

    for i in range(len(seq_srcs)):
        seq_srcs[i] = '[CLS] ' + seq_srcs[i] + ' [SEP]'

    cudics = pickle.load(open('./data/test_cudics.pkl', 'rb'))
    seq_tars = pickle.load(open('./data/test_tars.pkl', 'rb'))

    q_id = [i + 1 for i in range(len(dcmn_labels))]
    article = [u[0] for u in dcmn_srcs]
    question = [u[1] for u in dcmn_srcs]
    cts = []
    for i in range(config.num_choices):
        cts.append([u[i + 2] for u in dcmn_srcs])

    examples = [
        SwagExample(
            swag_id=s5,
            context_sentence=s1,
            start_ending=s2,
            endings=s3,
            label=s4,
        ) for i, (s1, s2, *s3, s4, s5) in
        enumerate(zip(article, question, *cts, dcmn_labels, q_id))
    ]

    features = convert_examples_to_features(examples, config.tokenizer, config.max_seq_length)
    input_ids = select_field(features, 'input_ids')
    input_mask = select_field(features, 'input_mask')
    segment_ids = select_field(features, 'segment_ids')
    doc_len = select_field(features, 'doc_len')
    ques_len = select_field(features, 'ques_len')
    option_len = select_field(features, 'option_len')
    labels = [f.label for f in features]

    dcmn_contents = []
    for i in range(len(input_ids)):
        dcmn_contents.append((input_ids[i], input_mask[i], segment_ids[i], doc_len[i], ques_len[i], option_len[i], labels[i]))

    seq_contents = []
    for i in range(len(seq_srcs)):
        seq_contents.append((seq_srcs[i], seq_tars[i], cudics[i], key_choices[i]))

    return seq_contents, dcmn_contents

