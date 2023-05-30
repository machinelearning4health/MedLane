
import numpy as np
from tqdm import trange
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import pickle
from transformers import BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score


def train_valid():
    MAX_LEN = 64
    bs = 64

    input_ids=pickle.load(open("./data/input_ids",'rb'))
    tags=pickle.load(open("./data/tags",'rb'))
    attention_masks=pickle.load(open("./data/attention_masks",'rb'))
    # attention_masks = [[int(v) for v in u] for u in attention_masks]
    tr_inputs = input_ids
    tr_tags = tags
    tr_masks = attention_masks

    input_ids=pickle.load(open("./data/test_input_ids",'rb'))
    tags=pickle.load(open("./data/test_tags",'rb'))
    attention_masks=pickle.load(open("./data/test_attention_masks",'rb'))

    val_inputs = input_ids
    val_tags = tags
    val_masks = attention_masks

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

    valid_data = TensorDataset(val_inputs, val_masks, val_tags)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

    tag_values = [0, 1]
    tag_values.append(2)
    tag2idx = {t: i for i, t in enumerate(tag_values)}

    model = BertForTokenClassification.from_pretrained(
        'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
        num_labels=len(tag2idx),
        output_attentions = False,
        output_hidden_states = False
    )
    model.cuda()
    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=5e-5,
        eps=1e-8
    )

    epochs = 3
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    ## Store the average loss after each epoch so we can plot them.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()

    loss_values, validation_loss_values = [], []

    for _ in trange(epochs, desc="Epoch"):
        # ========================================
        #               Training
        # ========================================
        # Perform one full pass over the training set.

        # Put the model into training mode.
        model.train()
        # Reset the total loss for this epoch.
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train_dataloader):
            # add batch to gpu
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            #         b_input_ids = b_input_ids.type(torch.LongTensor)
            #         b_input_mask= b_input_mask.type(torch.LongTensor)
            #         b_labels = b_labels.type(torch.LongTensor)

            #         b_input_ids = b_input_ids.to(device)
            #         b_input_mask = b_input_mask.to(device)
            #         b_labels = b_labels.to(device)
            b_input_ids = torch.tensor(b_input_ids).to(torch.int64)
            b_input_mask = torch.tensor(b_input_mask).to(torch.int64)
            b_labels = torch.tensor(b_labels).to(torch.int64)

            # Always clear any previously calculated gradients before performing a backward pass.
            model.zero_grad()
            # forward pass
            # This will return the loss (rather than the model output)
            # because we have provided the `labels`.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
            # get the loss
            loss = outputs[0]
            # Perform a backward pass to calculate the gradients.
            loss.backward()
            # track train loss
            total_loss += loss.item()
            # Clip the norm of the gradient
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            # update parameters
            optimizer.step()
            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)
        print("Average train loss: {}".format(avg_train_loss))

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        # Put the model into evaluation mode
        model.eval()
        # Reset the validation loss for this epoch.
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predictions, true_labels = [], []
        for batch in valid_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            b_input_ids = torch.tensor(b_input_ids).to(torch.int64)
            b_input_mask = torch.tensor(b_input_mask).to(torch.int64)
            b_labels = torch.tensor(b_labels).to(torch.int64)

            # Telling the model not to compute or store gradients,
            # saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have not provided labels.
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            # Move logits and labels to CPU
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            eval_loss += outputs[0].mean().item()
            predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(valid_dataloader)
        validation_loss_values.append(eval_loss)
        print("Validation loss: {}".format(eval_loss))
        pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                     for p_i, l_i in zip(p, l) if tag_values[l_i] != 2]

        valid_tags = [tag_values[l_i] for l in true_labels
                      for l_i in l if tag_values[l_i] != 2]

        print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
        print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
        print()

    torch.save(model, './cache/model.pth')
    model.cpu()


def main():
    train_valid()

if __name__ == '__main__':
    main()