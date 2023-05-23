import math
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import os
import pickle
from typing import List

class Vocabulary(object):

    def __init__(self):
        self.word2idx = {'<pad>': 0, '<cls>': 1, '<unk>': 2}
        self.idx2word = {0: '<pad>', 1: '<cls>', 2: '<unk>'}
        assert len(self.word2idx) == len(self.idx2word)
        self.idx = len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def vocab_construction(all_words, output_filename):
    vocab = Vocabulary()
    for word in all_words:
        vocab.add_word(word)
    print(f"Vocab len:", len(vocab))

    # sanity check
    assert set(vocab.word2idx.keys()) == set(vocab.idx2word.values())
    assert set(vocab.word2idx.values()) == set(vocab.idx2word.keys())
    for word in vocab.word2idx.keys():
        assert word == vocab.idx2word[vocab(word)]

    pickle.dump(vocab, open(output_filename, 'wb'))
    return


def build_vocab_mimic(data_dir):
    all_icu_stay_dict = pickle.load(open(os.path.join(data_dir, 'mimic_stay_dict.pkl'), 'rb'))
    all_codes = []
    for icu_id in all_icu_stay_dict.keys():
        for code in all_icu_stay_dict[icu_id].treatment:
            all_codes.append(code)
        for code in all_icu_stay_dict[icu_id].diagnosis:
            all_codes.append(code)
    all_codes = list(set(all_codes))
    vocab_construction(all_codes, os.path.join(data_dir, 'vocab.pkl'))


def to_index(sequence, vocab, prefix='', suffix=''):
    """ convert code to index """
    prefix = [vocab(prefix)] if prefix else []
    suffix = [vocab(suffix)] if suffix else []
    sequence = prefix + [vocab(token) for token in sequence] + suffix
    return sequence


class MIMICTokenizer:
    def __init__(self, data_dir):
        build_vocab_mimic(data_dir)
        self.vocab_dir = os.path.join(data_dir, 'vocab.pkl')
        if not os.path.exists(self.vocab_dir):
            build_vocab_mimic(data_dir)
        self.code_vocabs, self.code_vocabs_size = self._load_code_vocabs()
        self.type_vocabs, self.type_vocabs_size = self._load_type_vocabs()

    def _load_code_vocabs(self):

        vocabs = pickle.load(open(self.vocab_dir, 'rb'))
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def _load_type_vocabs(self):
        vocabs = Vocabulary()
        for word in ['dx', 'tr']:
            vocabs.add_word(word)
        vocabs_size = len(vocabs)
        return vocabs, vocabs_size

    def get_code_vocabs_size(self):
        return self.code_vocabs_size

    def get_type_vocabs_size(self):
        return self.type_vocabs_size

    def __call__(self,
                 batch_codes: List[str],
                 batch_types: List[str],
                 padding=True,
                 prefix='<cls>',
                 suffix=''):

        # to tensor
        batch_codes = [torch.tensor(to_index(c, self.code_vocabs, prefix=prefix, suffix=suffix)) for c in batch_codes]
        batch_types = [torch.tensor(to_index(t, self.type_vocabs, prefix=prefix, suffix=suffix)) for t in batch_types]

        # padding
        if padding:
            batch_codes = pad_sequence(batch_codes, batch_first=True)
            batch_types = pad_sequence(batch_types, batch_first=True)

        return batch_codes, batch_types


class Attention(nn.Module):
    def forward(self, query, key, value, mask, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = torch.softmax(scores, dim=-1)
        p_attn = p_attn.masked_fill(mask == 0, 0)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask):
        """
        :param query, key, value: [batch_size, seq_len, d_model]
        :param mask: [batch_size, seq_len, seq_len]
        :return: [batch_size, seq_len, d_model]
        """

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask.unsqueeze(1), dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x, mask):
        x = self.w_2(self.dropout(self.activation(self.w_1(x))))
        mask = mask.sum(dim=-1) > 0
        x[~mask] = 0
        return x


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """ Apply residual connection to any sublayer with the same size. """
        return x + self.dropout(sublayer(self.norm(x)))


class TransformerBlock(nn.Module):
    """
    Transformer Block = MultiHead Attention + Feed Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param dropout: dropout rate
        """

        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=4 * hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        print(f"TransformerBlock added with hid-{hidden}, head-{attn_heads}, in_hid-{2 * hidden}, drop-{dropout}")

    def forward(self, x, mask):
        """
        :param x: [batch_size, seq_len, hidden]
        :param mask: [batch_size, seq_len, seq_len]
        :return: batch_size, seq_len, hidden]
        """

        x = self.input_sublayer(x, lambda _x: self.attention(_x, _x, _x, mask=mask))
        x = self.output_sublayer(x, lambda _x: self.feed_forward(_x, mask=mask))
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, embedding_size: int, dropout: float, layers: int, heads: int, device='cpu'):
        super(Transformer, self).__init__()
        self.tokenizer = MIMICTokenizer("/data/")
        self.embedding_size = embedding_size
        self.dropout = dropout
        self.layers = layers
        self.heads = heads
        self.device = device

        # embedding
        self.code_embedding = nn.Embedding(self.tokenizer.get_code_vocabs_size(), embedding_size, padding_idx=0)
        self.type_embedding = nn.Embedding(self.tokenizer.get_type_vocabs_size(), embedding_size, padding_idx=0)

        # encoder
        self.transformer = nn.ModuleList([TransformerBlock(embedding_size, heads, dropout) for _ in range(layers)])

        # binary classifier
        self.fc = nn.Linear(embedding_size, 2)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        codes, types = x[0], x[1]
        codes, types = self.tokenizer(codes, types, padding=True, prefix='<cls>')
        codes = codes.cuda()
        types = types.cuda()

        """ embedding """
        # [# admissions, # batch_codes, embedding_size]
        codes_emb = self.code_embedding(codes)
        types_emb = self.type_embedding(types)
        emb = codes_emb + types_emb

        """ transformer """
        mask = (codes != 0)
        mask = torch.einsum('ab,ac->abc', mask, mask)
        for transformer in self.transformer:
            x = transformer(emb, mask)  # [# admissions, # batch_codes, embedding_size]

        cls_emb = x[:, 0, :]
        logits = self.fc(cls_emb)
        # logits = logits.squeeze(-1)
        return logits

    def get_cls_embed(self, x):
        codes, types = x[0], x[1]
        codes, types = self.tokenizer(codes, types, padding=True, prefix='<cls>')
        codes = codes.cuda()
        types = types.cuda()

        """ embedding """
        # [# admissions, # batch_codes, embedding_size]
        codes_emb = self.code_embedding(codes)
        types_emb = self.type_embedding(types)
        emb = codes_emb + types_emb

        """ transformer """
        mask = (codes != 0)
        mask = torch.einsum('ab,ac->abc', mask, mask)
        for transformer in self.transformer:
            x = transformer(emb, mask)  # [# admissions, # batch_codes, embedding_size]

        cls_embed = x[:, 0, :]  # get CLS embedding
        return cls_embed

def define_model_mimic(configuration):
    embedding_size = configuration["hidden_size"]
    dropout = 0.5
    layers = configuration["layer_num"]
    heads = configuration["head_num"]
    return Transformer(embedding_size, dropout, layers, heads)


class YearbookNetwork(nn.Module):

    def __init__(self, layers_dict):
        super(YearbookNetwork, self).__init__()
        self.enc = layers_dict["enc"]
        self.classifier = layers_dict["classifier"]

    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=(2, 3))
        x = self.classifier(x)
        return x


def define_model_yearbook(config):
    layers_dict = {}
    nn_space = config.copy()
    n_convs = 4
    pre_flat_size = 32
    in_channels = 3
    out_kernel = None
    layers = []
    for i in range(n_convs):
        if pre_flat_size > 7:
            out_channels = nn_space.get("n_conv_channels_c{}".format(i + 1))
            kernel_size = nn_space.get("kernel_size_c{}".format(i + 1))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            pre_flat_size = pre_flat_size - kernel_size+1
            if pre_flat_size > 3:
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool2d(2))
                pre_flat_size = int(pre_flat_size / 2)
        in_channels = out_channels
        out_kernel = kernel_size
    layers_dict["enc"] = nn.Sequential(*layers)
    layers = []
    in_features = out_channels
    layers.append(nn.Linear(in_features, 2))
    layers_dict["classifier"] = nn.Sequential(*layers)
    return YearbookNetwork(layers_dict)


import torch.nn as nn
from transformers import DistilBertModel, DistilBertForSequenceClassification


class DistilBertFeaturizer(DistilBertModel):
    def __init__(self, config):
        super().__init__(config)
        self.d_out = config.hidden_size

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        hidden_state = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )[0]
        pooled_output = hidden_state[:, 0]
        return pooled_output

class ArticleNetwork(nn.Module):
    def __init__(self, num_classes):
        super(ArticleNetwork, self).__init__()
        featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        classifier = nn.Linear(featurizer.d_out, num_classes)
        self.model = nn.Sequential(featurizer, classifier)

    def forward(self, x):
        return self.model(x)



from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

IMG_HEIGHT = 224
NUM_CLASSES = 62

class FMoWNetwork(nn.Module):
    def __init__(self):
        super(FMoWNetwork, self).__init__()
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)

    def forward(self, x):
        features = self.enc(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return self.classifier(out)