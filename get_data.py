from tqdm import tqdm
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import pickle
import os


MAX_LEN = 64
bs = 16  # batch_size
tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
with open('./data/abbrs-all-cased.pkl', 'rb') as f:
    abbrs = pickle.load(f)


def word_tokenize(sentence):
    tokens = tokenizer.tokenize(sentence)
    temp = ''
    for word in tokens:
        if word[0] == '#':
            word = word[2:]
            temp += word
        else:
            temp += ' '
            temp += word
    temp = temp.strip()

    return temp.split(' ')


def tokenize_and_preserve_labels(sentence, text_labels):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):
        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels


def get_mask(src_words, tar_words):
    if tar_words[-1] != '.':
        tar_words.append('.')
    i = 0
    j = 0
    mask = []

    while i < len(src_words):
        if j == len(tar_words):
            while i < len(src_words):
                mask.append(0)
                i += 1
            break

        if src_words[i] == tar_words[j]:
            mask.append(0)
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
            for k, word in enumerate(src_words[i:p]):
                key = word
                if key in abbrs.keys():
                    pass
                elif key.upper() in abbrs.keys():
                    key = key.upper()
                elif key.lower() in abbrs.keys():
                    key = key.lower()

                if key in abbrs.keys():
                    mask.append(1)
                else:
                    mask.append(0)

            i = p
            j = q

    return mask


def get_train_data(train_file_path):
    src = []
    tar = []
    txt = ''
    try:
        txt += open(train_file_path, 'r').read()
    except:
        txt += open(train_file_path, 'r', encoding='utf-8').read()

    txt = txt.split('\n\n')
    mask_new = []
    src_new = []
    for para in tqdm(txt,ncols=200):
        sentences = para.split('\n')
        if len(sentences) < 2:
            continue
        for sid, sentence in enumerate(sentences[0:2]):
            if sid == 0:
                src.append(sentence)
            else:
                tar.append(sentence)

    for i in range(len(src)):
        src_sentence = src[i]
        tar_sentence = tar[i]
        src_words = word_tokenize(src_sentence)
        tar_words = word_tokenize(tar_sentence)
        mask = get_mask(src_words, tar_words)
        src_new.append(src_words)
        mask_new.append(mask)
        assert len(src_words) == len(mask)

    return src_new, mask_new


def get_test_data(test_file_path):
    txt = open(test_file_path, 'r').read()
    #     txt = txt.lower()
    txt = txt.split('\n\n')
    mask = []
    src_new = []
    for para in tqdm(txt,ncols=200):
        sentences = para.split('\n')
        masks = []

        src_sentence = ''

        if len(sentences) < 2 or len(sentences[0]) < 3 or len(sentences[1]) < 3:
            continue
        for sid, sentence in enumerate(sentences):
            if sid == 0:
                src_sentence = sentence
                words = word_tokenize(sentence)
                src_new.append(words)

            elif sid == 1:
                cudic = {}
                sentence = sentence[2:].lower()
                text = re.sub('\[[^\[\]]*\]', '', sentence)
                pairs = re.findall('[^\[\] ]+\[[^\[\]]+\]', sentence)
                for pair in pairs:
                    pair = re.split('[\[\]]', pair)
                    cudic[pair[0]] = pair[1]
                words = word_tokenize(text)
                for wid, word in enumerate(words):
                    if word in cudic.keys():
                        words[wid] = cudic[word]
                new_text = ''
                for word in words:
                    new_text += word
                    new_text += ' '
                masks = get_mask(word_tokenize(src_sentence), word_tokenize(new_text))

        mask.append(masks)
    return src_new, mask

def generate_train(config):
    train_sentences, train_labels = get_train_data(os.path.join(config.data_dir, config.train_file))
    tag_values = [0, 1]
    tag_values.append(2)
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs)
        for sent, labs in zip(train_sentences, train_labels)
    ]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx[2], padding="post",
                         dtype="long", truncating="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
    pickle.dump(input_ids, open("./data/input_ids", 'wb'))
    pickle.dump(tags, open("./data/tags", 'wb'))
    pickle.dump(attention_masks, open("./data/attention_masks", 'wb'))

def generate_test(config):
    test_sentences, test_labels = get_test_data(os.path.join(config.data_dir, config.test_file))
    tag_values = [0, 1]
    tag_values.append(2)
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs)
        for sent, labs in zip(test_sentences, test_labels)
    ]
    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")
    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx[2], padding="post",
                         dtype="long", truncating="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    pickle.dump(input_ids, open("./data/test_input_ids", 'wb'))
    pickle.dump(tags, open("./data/test_tags", 'wb'))
    pickle.dump(attention_masks, open("./data/test_attention_masks", 'wb'))


def main():
    from config import DCMN_Config
    config = DCMN_Config()
    generate_train(config)
    generate_test(config)


if __name__ == '__main__':
    main()
