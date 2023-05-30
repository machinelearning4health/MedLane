# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        context = x[0]
        mask = x[1]
        bert_output = self.bert(context, attention_mask=mask)
        state = bert_output[0]
        pooled = bert_output[1]
        return state, pooled
