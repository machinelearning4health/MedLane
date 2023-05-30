import sys
import time
import logging
import os
import random
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from dcmn import BertForMultipleChoiceWithMatch
from train import train_valid
from config import DCMN_Config
from Seq2seq import DecoderRNN, Seq2seq, SEP, CLS
import models.bert as seq_bert
from dataset import build_dataset, build_dataset_eval
from utils import build_iterator, build_eval_iterator
import argparse
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def build_dcmn(config):

    output_eval_file = os.path.join(config.output_dir, config.output_file)

    if os.path.exists(output_eval_file) and config.output_file != 'output.txt':
        raise ValueError("Output file ({}) already exists and is not empty.".format(output_eval_file))
    with open(output_eval_file, "w") as writer:
        writer.write("***** Eval results Epoch  %s *****\t\n" % (
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        dic = str([(name, value) for name, value in vars(config).items()])
        writer.write("%s\t\n" % dic)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if not config.no_cuda > 0:
        torch.cuda.manual_seed_all(config.seed)

    train_seq_dataset, train_dcmn_dataset = build_dataset(config)
    # with open('./data/train_seq_dataset.pkl', 'wb') as f:
    #     pickle.dump(train_seq_dataset, f)
    # with open('./data/train_dcmn_dataset.pkl', 'wb') as f:
    #     pickle.dump(train_dcmn_dataset, f)
    # with open('./data/train_seq_dataset.pkl', 'rb') as f:
    #     train_seq_dataset = pickle.load(f)
    # with open('./data/train_dcmn_dataset.pkl', 'rb') as f:
    #     train_dcmn_dataset = pickle.load(f)
    train_dataloader = build_iterator(train_seq_dataset, train_dcmn_dataset, config)

    eval_seq_dataset, eval_dcmn_dataset = build_dataset_eval(config)
    # # with open('./data/eval_seq_dataset.pkl', 'wb') as f:
    # #     pickle.dump(eval_seq_dataset, f)
    # # with open('./data/eval_dcmn_dataset.pkl', 'wb') as f:
    # #     pickle.dump(eval_dcmn_dataset, f)
    # with open('./data/eval_seq_dataset.pkl', 'rb') as f:
    #     eval_seq_dataset = pickle.load(f)
    # with open('./data/eval_dcmn_dataset.pkl', 'rb') as f:
    #     eval_dcmn_dataset = pickle.load(f)
    eval_dataloader = build_eval_iterator(eval_seq_dataset, eval_dcmn_dataset, config)

    num_train_steps = int(
        len(train_seq_dataset) / config.batch_size / config.gradient_accumulation_steps * (config.num_train_epochs-config.num_dcmn_epochs))
    t_total = num_train_steps
    config.t_total = t_total

    dcmn_t_total = 0
    for step, (seq_batches, dcmn_batches) in enumerate(train_dataloader):
        if len(dcmn_batches) > 0:
            dcmn_t_total += len(dcmn_batches) // config.batch_size
            if len(dcmn_batches) % config.batch_size > 0:
                dcmn_t_total += 1
    dcmn_t_total *= config.num_dcmn_epochs

    model = BertForMultipleChoiceWithMatch.from_pretrained(config.bert_model, num_choices=config.num_choices)
    model.to(config.dcmn_device)

    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=config.dcmn_learning_rate,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config.dcmn_warmup_proportion * dcmn_t_total),
                                                num_training_steps=dcmn_t_total)  # PyTorch scheduler

    loss_fun = torch.nn.CrossEntropyLoss()

    return model, train_dataloader, eval_dataloader, optimizer, scheduler, loss_fun


def build_seq2seq(config, hidden_size, no_cuda):
    bidirectional = False

    config.hidden_size = hidden_size

    encoder = seq_bert.Model(config).to(config.seq_device)
    decoder = DecoderRNN(len(config.tokenizer.vocab), config.max_seq_length,
                         hidden_size * 2 if bidirectional else hidden_size,
                         dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                         eos_id=config.tokenizer.convert_tokens_to_ids([SEP])[0],
                         sos_id=config.tokenizer.convert_tokens_to_ids([CLS])[0])

    decoder = decoder.to(config.seq_device)
    seq2seq = Seq2seq(encoder, decoder)
    seq2seq = seq2seq.to(config.seq_device)
    param_optimizer = list(seq2seq.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    optimizer = AdamW(params=optimizer_grouped_parameters,
                      lr=config.learning_rate,
                      correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(config.warmup_proportion * config.t_total),
                                                num_training_steps=config.t_total)  # PyTorch scheduler

    loss_fun = torch.nn.NLLLoss(reduction='none')

    return seq2seq, optimizer, scheduler, loss_fun


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_step1",
                        action="store_true",
                        help="whether to run step1")
    parser.add_argument("--bert_path",
                        default='./cache/model.pth',
                        type=str,
                        help="bert path for step1")
    parser.add_argument("--para_path",
                        default=None,
                        type=str,
                        help="para path for dcmn and seq2seq")

    parser.add_argument("--data_path",
                        default='./data',
                        type=str,
                        help="dataset path")
    parser.add_argument("--train_file",
                        default='train(12809).txt',
                        type=str,
                        help="filename of train dataset")
    parser.add_argument("--test_file",
                        default='test(2030).txt',
                        type=str,
                        help="filename of test dataset")


    args = parser.parse_args()
    config = DCMN_Config()
    config.data_dir = args.data_path
    config.train_file = args.train_file
    config.test_file = args.test_file

    if not args.skip_step1:
        from get_data import generate_test, generate_train
        generate_train(config)
        generate_test(config)
        import step1
        step1.train_valid()

    from get_mask import get_mask
    get_mask(args.bert_path, config)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    dcmn, train_dataloader, eval_dataloader, dcmn_optimizer, dcmn_scheduler, dcmn_loss_fun = build_dcmn(config)

    seq2seq, seq_optimizer, seq_scheduler, seq_loss_fun = build_seq2seq(config, 768, config.no_cuda)

    if args.para_path is not None:
        save_file_best = torch.load(args.para_path, map_location=torch.device('cuda:0'))
        dcmn.load_state_dict(save_file_best['dcmn_para'])
        seq2seq.load_state_dict(save_file_best['seq_para'])

    # dcmn = torch.nn.DataParallel(dcmn)
    # dcmn = dcmn.cuda()
    # seq2seq = torch.nn.DataParallel(seq2seq)
    # seq2seq = seq2seq.cuda()

    train_valid(dcmn, config, train_dataloader, eval_dataloader, dcmn_optimizer, dcmn_scheduler, dcmn_loss_fun,
                seq2seq, seq_optimizer, seq_scheduler, seq_loss_fun)


if __name__ == '__main__':
    main()
