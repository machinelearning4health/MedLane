import os
import time
import torch
from transformers import BertTokenizer


class DCMN_Config():

    def __init__(self):
        # The input data dir. Should contain the .csv files (or other data files) for the task.
        self.data_dir = './data'
        self.train_file = 'train(sample).txt'
        self.test_file = 'test(sample).txt'

        # Bert pre-trained model selected in the list: bert-base-uncased,
        # bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.
        self.bert_model = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

        # The output directory where the model checkpoints will be written.
        self.output_dir = 'result'
        self.output_file = 'output.txt'

        self.max_seq_length = 64
        self.batch_size = 3
        self.eval_batch_size = 1
        self.num_choices = 8
        self.learning_rate = 5e-5
        self.dcmn_learning_rate = 1e-5
        self.num_train_epochs = 36.0
        self.num_dcmn_epochs = 6.0
        self.model_name = 'output_dcmn.bin'


        # Number of updates steps to accumulate before performing a backward/update pass.
        self.gradient_accumulation_steps = 1

        # Proportion of training to perform linear learning rate warmup for.
        # E.g., 0.1 = 10%% of training.
        self.warmup_proportion = 0.03
        self.dcmn_warmup_proportion = 0.1

        # Whether not to use CUDA when available
        self.no_cuda = False

        # random seed for initialization
        self.seed = 42

        # Whether to perform optimization and keep the optimizer averages on CPU
        self.optimize_on_cpu = False

        # Loss scaling, positive power of 2 values can improve fp16 convergence.
        self.loss_scale = 4

        if self.no_cuda:
            self.dcmn_device = torch.device("cpu")
            self.seq_device = torch.device("cpu")
        else:
            # torch.cuda.set_device(self.gpu_id)
            self.dcmn_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.seq_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.gradient_accumulation_steps < 1:
            raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                self.gradient_accumulation_steps))

        self.batch_size = int(self.batch_size / self.gradient_accumulation_steps)

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        self.t_total = 0
        self.hidden_size = 768


if __name__ == '__main__':
    config = DCMN_Config()
