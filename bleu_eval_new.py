import pickle
from nltk.translate.bleu_score import sentence_bleu
#pre_dic = pickle.load(open('data/test_dic.pkl', 'rb'))
import pickle
import random
import os
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from nlgeval import compute_individual_metrics, compute_metrics
import numpy as np
from dataset import word_tokenize
from tqdm import tqdm

words = stopwords.words('english')
wordnet_lemmatizer = WordNetLemmatizer()

common_words = open('./data/common.txt', 'r').readlines()
common_words = [re.sub('\n', '', w) for w in common_words]
common_words = set(common_words+words)
pro_dic = pickle.load(open('./data/exist_dic_new.pkl', 'rb'))
pro_words = set(pro_dic.keys())


def obtain(txt, exist_dic, src, tar, config):
    for para in txt:
        sentences = para.split('\n')
        tars = []
        dics = []

        if len(sentences) < 2 or len(sentences[0]) < 3 or len(sentences[1]) < 3:
            continue
        for sid, sentence in enumerate(sentences):
            if sid == 0:
                src.append(sentence + ' EOST')
            else:
                cudic = {}
                sentence = sentence[2:]
                text = re.sub('\[[^\[\]]*\]', '', sentence)
                pairs = re.findall('[^\[\] ]+\[[^\[\]]+\]', sentence)
                for pair in pairs:
                    pair = re.split('[\[\]]', pair)
                    cudic[pair[0]] = pair[1]
                words = word_tokenize(text, config.tokenizer)
                for wid, word in enumerate(words):
                    if word in cudic.keys():
                        words[wid] = cudic[word]
                new_text = ''
                for word in words:
                    new_text += word
                    new_text += ' '
                tars.append(new_text)
                dics.append(cudic)

        tar.append(tars)
        exist_dic.append(dics)


def lemmatize(word, tag):
    if tag.startswith('NN'):
        return wordnet_lemmatizer.lemmatize(word, pos='n')
    elif tag.startswith('VB'):
        return wordnet_lemmatizer.lemmatize(word, pos='v')
    elif tag.startswith('JJ'):
        return wordnet_lemmatizer.lemmatize(word, pos='a')
    elif tag.startswith('R'):
        return wordnet_lemmatizer.lemmatize(word, pos='r')
    else:
        return word

def mark_sentence(sentence):
    sentence = sentence.lower()
    count_pro = 0
    count_unc = 0
    count_total = 0
    sentence = re.sub(r'-?\d+\.?\d*e?-?\d*?', ' num ', sentence)
    words = nltk.word_tokenize(sentence.lower())
    tag = nltk.pos_tag(words)
    for wid, word in enumerate(words):
        word = lemmatize(word, tag[wid][1])
        count_total += 1
        if word in pro_words:
            words[wid] = 'PRO'
            count_pro += 1
        else:
            if word not in common_words and word.isalpha():
                words[wid] = 'UNCOMMON'
                count_unc += 1
            else:
                words[wid] = word
        if words[wid]== 'num':
            words[wid] = 'NUM'
    return count_unc, count_pro, count_total

# def replace_pro(sentence):
#     sentence = sentence.split(' ')
#     tar = ''
#     for wid, word in enumerate(sentence):
#         if word in pre_dic.keys():
#             sentence[wid] = pre_dic[word]
#     for word in sentence:
#         if word != sentence[-1]:
#             tar += (word+' ')
#         else:
#             tar += word
#     return tar

def contact_word_spilt(sentence):
    sentence = re.sub('@@ ', '', sentence)
    #sentence = replace_pro(sentence)
    sentence = sentence.split(' ')
    return sentence

def get_sentence_bleu(candidate, reference):
    score = sentence_bleu(reference, candidate)
    return score


def count_score(candidate, reference):
    avg_score = 0
    for k in range(len(candidate)):
        reference_ = reference[k]
        for m in range(len(reference_)):
            reference_[m] = nltk.word_tokenize(reference_[m])
        candidate[k] = nltk.word_tokenize(candidate[k])
        try:
            tmp = get_sentence_bleu(candidate[k], reference_)
            if tmp < 0.2:
                print(' '.join(reference_[0]))
            avg_score += tmp/len(candidate)
        except:
            print(candidate[k])
            print(reference[k])
    return avg_score

def count_hit(candidate, dics):
    avg_score = 0
    for sentence, cdics in zip(candidate, dics):
        max_score = 0
        for cdic in cdics:
            words = sentence
            txt = ''
            for word in words:
                txt += word
                txt += ' '
            count = 0
            for value in cdic.values():
                rs = re.findall(value, txt)
                if len(rs) > 0:
                    count += 1
            if len(cdic) == 0:
                score = 1.0
            else:
                score = count/len(cdic)
            if score > max_score:
                max_score = score
        avg_score += max_score/len(candidate)
    return avg_score

def count_common(candidate):
    avg_score = 0
    for sentence in candidate:
        txt = ''
        for word in sentence:
            txt += word
            txt += ' '
        txt = txt[0:-1]
        unc, pro, count = mark_sentence(txt)
        coms = (count-unc-pro) / (count+1e-3)
        avg_score += coms/len(candidate)
    return avg_score


def count_feature_score(candidates):
    unss = []
    pross = []
    for sentence in candidates:
        unc, pro, count = mark_sentence(sentence)
        uns = unc / count
        pros = pro / count
        unss.append(uns)
        pross.append(pros)
    unss = np.array(unss)
    pross = np.array(pross)
    return 1-unss.mean()-pross.mean()



def get_score(config, is_val=True):
    results = open('./result/tmp.out.txt', 'r', encoding='utf-8').readlines()

    txt = open(os.path.join(config.data_dir, config.test_file), 'r').read()
    txt = txt.lower()
    txt = txt.split('\n\n')
    if is_val:
        txt = txt[0:len(txt) // 2]
        results = results[0:len(results)//2]
    else:
        txt = txt[len(txt) // 2:]
        results = results[len(results)//2:]
    src = []
    tar = []
    exist_dic = []
    obtain(txt, exist_dic, src, tar, config)
    for u in tar:
        if len(u)<4:
            print(u)

    pickle.dump(exist_dic, open('./data/test_dic.pkl', 'wb'))
    pickle.dump(src, open('./data/test.pro.pkl', 'wb'))
    pickle.dump(tar, open('./data/test.cus.pkl', 'wb'))

    sources = pickle.load(open('./data/test.pro.pkl', 'rb'))
    sources = [x.replace('\n', '') for x in sources]
    ref = pickle.load(open('./data/test.cus.pkl', 'rb'))
    dics = pickle.load(open('./data/test_dic.pkl', 'rb'))
    test_subjects = np.array(results)
    test_targets = np.array(ref)
    test_dics = np.array(dics)
    len_sen = [len(nltk.word_tokenize(x)) for x in sources]
    len_sen = np.array(len_sen)
    # print(len_sen.mean(), len_sen.max(), len_sen.min())
    len_spilt = [(0, 100000)]

    for len_current in len_spilt:
        index = np.where((len_sen >= len_current[0]) & (len_sen < len_current[1]))
        ref = test_targets[index].tolist()
        hyp = test_subjects[index].tolist()
        open('./tmp/hyp.txt', 'w', encoding='utf-8').writelines([x for x in hyp])
        ref0 = [x[0] for x in ref]
        ref1 = [x[1] for x in ref]
        ref2 = [x[2] for x in ref]
        ref3 = [x[3] for x in ref]
        open('./tmp/ref0.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref0])
        open('./tmp/ref1.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref1])
        open('./tmp/ref2.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref2])
        open('./tmp/ref3.txt', 'w', encoding='utf-8').writelines([x + '\n' for x in ref3])

        dics = test_dics[index].tolist()

        metrics_dict = compute_metrics(hypothesis='./tmp/hyp.txt',
                                       references=['./tmp/ref0.txt', './tmp/ref1.txt', './tmp/ref2.txt', './tmp/ref3.txt'],
                                       no_glove=True, no_overlap=False, no_skipthoughts=True)
        hyp = [nltk.word_tokenize(x) for x in hyp]
        hit = count_hit(hyp, dics)

        com = count_common(hyp)
        BLEU = (metrics_dict['Bleu_1'] + metrics_dict['Bleu_2'] + metrics_dict['Bleu_3'] + metrics_dict['Bleu_4']) / 4
        if BLEU<0.0001:
            BLEU = 0.0001
        if hit<0.0001:
            hit = 0.0001
        if com<0.0001:
            com = 0.0001
        Ascore = (1 + 2.25 + 4) / (4 / BLEU + 2.25 / hit + 1 / com)
        return BLEU, hit, com, Ascore


if __name__ == '__main__':
    from config import DCMN_Config
    config = DCMN_Config()
    config.test_file = 'test(2030).txt'
    get_score(config)