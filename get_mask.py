import numpy as np
import torch
from transformers import BertTokenizer
import os

import pickle


def softmax(x, axis=1):
    row_max = x.max(axis=axis)

    row_max = row_max.reshape(-1, 1)
    x = x - row_max

    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def get_mask(model_path, config):
    model = torch.load(model_path)
    tag_values = [0, 1]
    tag_values.append(2)
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    # generate the dataset for dcmn and seq2seq
    txt = open(os.path.join(config.data_dir, config.test_file), 'r').read()
    # txt = txt.lower()
    txt = txt.split('\n\n')
    test_sentences = []
    for u in txt:
        sts = u.split('\n')
        test_sentences.append(sts[0])

    tokenizer = config.tokenizer

    test_sts_step2 = []
    test_mask_step2 = []
    from tqdm import tqdm

    for test_sentence in tqdm(test_sentences,ncols=200):
        tokenized_sentence = tokenizer.encode(test_sentence)
        input_ids = torch.tensor([tokenized_sentence]).cuda()

        with torch.no_grad():
            output = model(input_ids)
        logits = output[0].to('cpu').numpy()[0]
        prob = softmax(logits)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        # join bpe split tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        new_prob = []
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(tag_values[label_idx])
                new_tokens.append(token)
                new_prob.append(prob[label_idx][1])

        new_tokens = new_tokens[1:-1]
        new_labels = new_labels[1:-1]
        test_sts_step2.append(new_tokens)
        test_mask_step2.append(new_labels)

    with open('./data/test_mask_step2_2030.pkl', 'wb') as f:
        pickle.dump(test_mask_step2, f)


def main():
    from config import DCMN_Config
    config = DCMN_Config()
    get_mask('./cache/model.pth',config)


if __name__ == '__main__':
    main()