import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np
# from utils import decode_sentence

PAD, CLS = '[PAD]', '[CLS]'
SEP = '[SEP]'

class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim*2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
        mix = torch.bmm(attn, context)

        # concat -> (batch, out_len, 2*dim)
        combined = torch.cat((mix, output), dim=2)
        # output -> (batch, out_len, dim)
        output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn


class BaseRNN(nn.Module):

    SYM_MASK = "MASK"
    SYM_EOS = "EOS"

    def __init__(self, vocab_size, max_len, hidden_size, input_dropout_p, dropout_p, n_layers, rnn_cell):
        super(BaseRNN, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.input_dropout_p = input_dropout_p
        self.input_dropout = nn.Dropout(p=input_dropout_p)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.dropout_p = dropout_p

    def forward(self, *args, **kwargs):
        raise NotImplementedError()


class DecoderRNN(BaseRNN):

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False):
        super(DecoderRNN, self).__init__(vocab_size, max_len, hidden_size,
                                         input_dropout_p, dropout_p,
                                         n_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size, hidden_size, n_layers, batch_first=True, dropout=dropout_p)
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        if use_attention:
            self.attention = Attention(self.hidden_size)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.hidden_change = nn.Linear(768, self.hidden_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size,
                                                                                                           output_size,
                                                                                                           -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, training=True):
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        device = encoder_hidden.get_device()
        if device>=0:
            inputs = inputs.to(torch.device('cuda:{}'.format(device)))

        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols) + 1
            return symbols

        if training:
            # Manual unrolling is used to support random teacher forcing.
            # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
            if use_teacher_forcing:
                decoder_input = inputs[:, :-1]
                decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                         function=function)

                for di in range(decoder_output.size(1)):
                    step_output = decoder_output[:, di, :]
                    if attn is not None:
                        step_attn = attn[:, di, :]
                    else:
                        step_attn = None
                    symbols = decode(di, step_output, step_attn)
                    sequence_symbols.append(symbols)
            else:
                decoder_input = inputs[:, 0].unsqueeze(1)
                for di in range(max_length):
                    decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                                  encoder_outputs,
                                                                                  function=function)
                    step_output = decoder_output[:, 0, :]
                    symbols = decode(di, step_output, step_attn)
                    sequence_symbols.append(symbols)
                    decoder_input = symbols
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                          encoder_outputs,
                                                                          function=function)
            decoder_output_pre = None
            sid = 1
            generation_state = False
            while sid < inputs.shape[1]:
                if inputs[0][sid] == 4:
                    generation_state = not generation_state

                if not generation_state:
                    decoder_input = inputs[:, sid].unsqueeze(1)
                    decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                                  encoder_outputs,
                                                                                  function=function)
                    if inputs[0][sid] != 4:
                        sequence_symbols.append(decoder_input)
                    sid += 1
                else:
                    decoder_input = inputs[:, sid].unsqueeze(1)
                    sequence_symbols.append(decoder_input)
                    decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                                  encoder_outputs,
                                                                                  function=function)
                    step_output = decoder_output.squeeze(1)
                    symbols = decode(0, step_output, step_attn)
                    sequence_symbols.append(symbols)
                    decoder_input = symbols
                    di = 1
                    while di==1 or (di < 20 and symbols[0][0] != 4):
                        decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                                      encoder_outputs,
                                                                                      function=function)
                        step_output = decoder_output.squeeze(1)
                        symbols = decode(di, step_output, step_attn)
                        sequence_symbols.append(symbols)
                        decoder_input = symbols
                        di += 1
                    if di == 20:
                        sequence_symbols.append(torch.LongTensor([[4]]).to(torch.device('cuda:0')))
                    sid += 1
                    while sid < inputs.shape[1] and inputs[0, sid] != 4:
                        sid += 1

            #symbols = sequence_symbols
            #symbols = torch.cat(symbols, 1).data.cpu().numpy()
            #result_temp = decode_sentence(symbols, config)
            #if result_temp[0:10] == ' however ,':
            #    print('here')
            #print(result_temp)


        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        encoder_hidden = torch.tanh(self.hidden_change(encoder_hidden)).unsqueeze(dim=0)
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(0)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(0)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


class Seq2seq(nn.Module):

    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, batch_src, batch_tar=None,
                teacher_forcing_ratio=0, training=True):

        encoder_outputs, encoder_hidden = self.encoder(batch_src)


        target_variable = batch_tar
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              # encoder_outputs=combined_embeddings,
                              encoder_outputs = encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio,
                              training=training)
        return result
