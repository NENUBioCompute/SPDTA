import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from models.encoder import Encoder
from torch.autograd import Variable

# GINConv model
from models.interformer import *


d_model = 128 #Embedding Size
d_k = d_v = 32 #dimension of K(=Q), V
n_heads = 3 #number of heads in Multi-Head Attention



class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input: torch.Tensor):

        return input.squeeze(self.dim)


class CDilated(nn.Module):

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        output = self.conv(input)
        return output


class DilatedParllelResidualBlockB(nn.Module):

    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())

        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        # reduce
        output1 = self.c1(input)
        output1 = self.br1(output1)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)

        # heirarchical fusion for de-gridding
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8

        # merge
        combine = torch.cat([d1, add1, add2, add3], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.br2(combine)
        return output

class SMILES_CONV(nn.Module):

    def __init__(
            self, size=70, num_tasks=1,
            hidden=128, num_layers=2,
            fc_hidden=-1, use_fp=False, use_desc=False, **kwargs
    ):
        super().__init__()
        # layers for smiles
        assert use_fp in [True, False]
        self.use_desc = use_desc
        self.use_fp = use_fp
        if fc_hidden == -1:
            fc_hidden = 128
        self.fc_hidden = fc_hidden
        self.embed_smi = nn.Embedding(size, hidden//2)
        conv_smi = []


        conv_smi.append(DilatedParllelResidualBlockB(hidden//2, hidden))
        for i in range(num_layers - 1):
            conv_smi.append(DilatedParllelResidualBlockB(hidden, hidden))
        # conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze(-1))
        self.conv_smi = nn.Sequential(*conv_smi)
        self.num_tasks = num_tasks
        #
        lin1_input = hidden
        if self.use_fp and self.use_desc:
            lin1_input = hidden + 2727 + 151
        elif not self.use_fp and self.use_desc:
            lin1_input = hidden + 151
        elif self.use_fp and not self.use_desc:
            lin1_input = hidden + 2727


        #lstm
        self.bilstm = nn.LSTM(hidden, hidden, num_layers=num_layers-1, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveMaxPool1d(1)


        self.lin1 = nn.Sequential(
            nn.Linear(lin1_input*2, fc_hidden),
            # nn.Dropout(0.5),
            nn.Dropout(0.1),
            nn.PReLU(),
        )


    def forward(self, smiles):

        # print('smi_tensors', smi_tensors.shape)
        smi_embed = self.embed_smi(smiles)  # (N,L,32)
        smi_embed = torch.transpose(smi_embed, 1, 2)

        smi_conv = self.conv_smi(smi_embed)  # (N,128,L)

        x = smi_conv.permute(0,2,1)

        # out, (h_n, _) = self.bilstm(smi_conv.permute(0,2,1))
        # out = self.pool(out.permute(0, 2, 1)).permute(0, 2, 1)

        # out = self.pool(smi_conv).permute(0, 2, 1)
        # x = out
        # print(out.shape)

        # x = self.lin1(out)

        return x
class BiLstm(nn.Module):
    def __init__(self,size, hidden, num_layers):
        super().__init__()
        self.embed_smi = nn.Embedding(size, hidden)
        self.bilstm = nn.LSTM(hidden, hidden, num_layers=num_layers - 1, batch_first=True, bidirectional=True)
        self.lin1 = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            # nn.Dropout(0.5),
            nn.Dropout(0.1),
            nn.PReLU(),
        )

        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, data):
        embed = self.embed_smi(data)  # (N,L,128)
        out, (h_n, _) = self.bilstm(embed)
        # out = self.pool(out.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.lin1(out)

        return x




device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")


class atten(nn.Module):
    def __init__(self, padding=3):
        super(atten, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(2 * padding + 1), padding=padding,
                               bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        xc = x.unsqueeze(1)
        xt = torch.cat((xc, self.dropout(xc)), dim=1)
        att = self.sigmoid(self.conv1(xt))
        return att.squeeze(1)


class LinkAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_first = torch.nn.Linear(128, 32)
        self.linear_second = torch.nn.Linear(32, 10)

    def softmax(self, input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x):
        sentence_att = F.tanh(self.linear_first(x))
        sentence_att = self.linear_second(sentence_att)
        sentence_att = self.softmax(sentence_att, 1)
        sentence_att = sentence_att.transpose(1, 2)
        sentence_embed = sentence_att @ x
        avg_sentence_embed = torch.sum(sentence_embed, 1) / 10

        return avg_sentence_embed


class FModel(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(FModel, self).__init__()
        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers

        self.merge_atten = atten(3)
        self.merge_atten2 = atten(3)

        self.down_sample1 = nn.Linear(256, 128)
        self.down_sample2 = nn.Linear(256, 128)

        self.activation = nn.ReLU()

        self.WC2 = nn.Linear(128, 128)
        self.WW = nn.Linear(128, 128)

        self.WF = nn.Linear(128, 128)
        self.WS = nn.Linear(128, 128)

        self.WC = nn.Linear(128, 128)
        self.WM = nn.Linear(128, 128)

        self.WC3 = nn.Linear(128, 128)
        self.WM3 = nn.Linear(128, 128)

        self.softmax1 = nn.Softmax(dim=1)


        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)




        # combined layers
        self.fc1 = nn.Linear(288, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task


        #pool
        self.adaptpool = nn.AdaptiveAvgPool2d((1,10))


        #mol2vec
        self.molconv1 = nn.Conv1d(1,1,1,1)
        self.molconv2 = nn.Conv1d(1,1,1,1)
        self.glu = nn.GLU()
        # self.adapt_max_pool = nn.AdaptiveMaxPool1d(128)
        # self.pool = nn.AdaptiveAPool1d(1)

        self.decoder = Decoder(128, 128, 1, 4, 128,DecoderLayer,SelfAttention, PositionwiseFeedforward, 0.2)




        # self.molprocess = MolProcess(32, 128, 3, 7, 0.1, device)

        #layer for ss
        # self.struc_process= BiLstm(size=4, hidden=128, num_layers=2)
        #
        # #layer for smiles
        # self.smiles_process = SMILES_CONV()
        self.struc_process = SMILES_CONV(size=4)
        # self.seq_process = SMILES_CONV(sizie=num_features_xt + 1)
        self.smiles_process = BiLstm(size=71, hidden=128, num_layers=2)
        self.seq_process = BiLstm(size=num_features_xt + 1, hidden=128, num_layers=2)

        self.link_attention = LinkAttention()




        self.fc = nn.Linear(256, 128)

        self.lin1 = nn.Sequential(
            nn.Linear(160, 128),
            # nn.Dropout(0.5),
            nn.Dropout(0.2),
            nn.PReLU(),
        )

        #fcfp
        self.fcfp_lin = nn.Sequential(
            nn.Linear(2048,1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )


        # dense layer
        self.lin = nn.Sequential(
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 1)

        )

    def Elem_feature_Fusion_D(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample1(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten(x_c)
        xs_ = self.activation(self.WC2(xs))
        x_ = self.activation(self.WW(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys

    def Elem_feature_Fusion_P(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        x_c = self.down_sample2(torch.cat((xs, x), dim=2))
        x_c = self.merge_atten2(x_c)
        xs_ = self.activation(self.WF(xs))
        x_ = self.activation(self.WS(x))
        xs_m = x_c * xs_ + xs
        ones = torch.ones(x_c.shape).to(device)
        x_m = (ones - x_c) * x_ + x
        ys = xs_m + x_m
        return ys

    def attention_PM(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.activation(self.WC(xs))
        h = self.activation(self.WM(x))
        weights = torch.matmul(xs_, h.permute(0, 2, 1))
        scale = weights.size(1) ** -0.5
        ys = self.softmax1(torch.matmul(weights, h) * scale) * xs
        return ys

    def attention_PM2(self, xs, x):
        """The attention mechanism is applied to the last layer of CNN."""
        xs_ = self.activation(self.WC3(xs))
        h = self.activation(self.WM3(x))
        weights = torch.matmul(xs_, h.permute(0, 2, 1))
        scale = weights.size(1) ** -0.5
        ys = self.softmax1(torch.matmul(weights, h) * scale) * xs
        return ys

    def make_masks(self, batch, compound_max_len, protein_max_len):
        batch_size = batch
        compound_mask = torch.zeros((batch_size, compound_max_len))
        protein_mask = torch.zeros((batch_size, protein_max_len))

        for i in range(batch_size):
            compound_mask[i, :103] = 1
            protein_mask[i, :1200] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(2)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2)
        return compound_mask, protein_mask

    def forward(self, data):

        target = data.target
        target_struc = data.target_struc

        # mol2vec = data.mol2vec
        compound_words = data.compound_words
        morgan = data.morgan


        compound_vector = self.smiles_process(compound_words)

        morgan_vector = morgan.view(-1, 1, 2048).float()
        mol_vector = self.fcfp_lin(morgan_vector)


        compound_vectors = self.dropout(self.attention_PM(compound_vector, mol_vector))  # .permute(0, 2, 1)
        mol_vectors = self.dropout(self.attention_PM2(mol_vector, compound_vector))

        compound_word_att = torch.cat((compound_vector, mol_vector), 1)
        mol_FCFPs_att = torch.cat((compound_vectors, mol_vectors), 1)


        drug_vectors = self.Elem_feature_Fusion_D(compound_word_att, mol_FCFPs_att)

        xd = self.link_attention(drug_vectors).squeeze()


        prot_seq = self.seq_process(target)

        prot_struc = self.struc_process(target_struc)

        target_vectors = self.Elem_feature_Fusion_P(prot_seq, prot_struc)

        xt = self.link_attention(target_vectors).squeeze()


        xc = torch.cat((xd,xt), 1)


        #add some dense layers
        out = self.lin(xc)

        return out