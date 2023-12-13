import torch
import torch.nn as nn
import torch.nn.functional as F


class layer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class encoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.first = layer(d_v, hidden_states[0], dropout)
        self.mid = nn.ModuleList([layer(hidden_states[_], hidden_states[_+1], dropout) for _ in range(n_layer-1)])
        self.last = nn.Linear(hidden_states[n_layer-1], d_emb)
        self.bn = nn.BatchNorm1d(d_emb)

    def forward(self, x):
        x = self.first(x)
        for layer in self.mid:
            x = layer(x)
        x = self.last(x)
        x = self.bn(x)
        return x


class decoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.first = layer(d_emb, hidden_states[n_layer-1], dropout)
        self.mid = nn.ModuleList([layer(hidden_states[n_layer-1-_], hidden_states[n_layer-2-_], dropout) for _ in range(n_layer-1)])
        self.last = nn.Linear(hidden_states[0], d_v)

    def forward(self, x):
        x = self.first(x)
        for layer in self.mid:
            x = layer(x)
        x = self.last(x)
        return x


class mid(nn.Module):
    def __init__(self, d_emb, n_block, n_view, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([layer(d_emb, d_emb, dropout) for _ in range(n_block)])
        self.bn = nn.BatchNorm1d(n_view)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.bn(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_emb, n_cls, dropout):
        super().__init__()
        self.linear = nn.Linear(d_emb, round(0.5 * d_emb))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(round(0.5 * d_emb), n_cls)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class ModelFirst(nn.Module):
    def __init__(self, d_v, n_cls, theta, dropout=0):
        super().__init__()
        self.first = layer(d_v, 2 * int(round(theta * d_v)), dropout)
        self.last = nn.Linear(int(round(theta * d_v)), n_cls)
        self.sigmoid = nn.Sigmoid()
        self.dim = int(round(theta * d_v))

    def forward(self, x, mask, flag=0):
        # mask: B * 1
        x = x * mask
        if not flag:
            statistics = self.first(x)
            mu = statistics[:, :self.dim]
            std = F.softplus(statistics[:, self.dim:]-5, beta=1)
            first_hidden = self.reparametrize(mu, std) * mask
        else:
            mu, std = None, None
            first_hidden = self.first(x)[:, :self.dim]

        x = self.last(first_hidden)
        x = self.sigmoid(x)
        x = x * mask
        return x, first_hidden, (mu, std)

    def reparametrize(self, mu, std):
        # reference :
        # http://pytorch.org/docs/0.3.1/_modules/torch/distributions.html#Distribution.sample_n
        eps = torch.zeros(std.size()).normal_().cuda('cuda:0')

        return mu + eps * std


class AutoEncoder(nn.Module):
    def __init__(self, r_list, d_emb, n_enc_layer, n_dec_layer, dropout=0.1):
        super().__init__()
        n_view = len(r_list)
        enc_hidden_states = []
        dec_hidden_states = []

        for _ in range(n_view):
            temp_hidden_states = []
            temp_hidden_states_ = []
            for i in range(n_enc_layer):
                hd = round(d_emb * 0.8)
                hd = int(hd)
                temp_hidden_states.append(hd)
            for i in range(n_dec_layer):
                hd = round(d_emb * 0.8)
                hd = int(hd)
                temp_hidden_states_.append(hd)

            enc_hidden_states.append(temp_hidden_states)
            dec_hidden_states.append(temp_hidden_states_)

        self.encoder_list = nn.ModuleList([encoder(r_list[v], enc_hidden_states[v], d_emb, n_enc_layer, dropout) for v in range(n_view)])
        self.decoder_list = nn.ModuleList([decoder(r_list[v], dec_hidden_states[v], d_emb, n_dec_layer, dropout) for v in range(n_view)])

        self.mid = mid(d_emb, 1, n_view, dropout)

        self.n_view = n_view
        self.r_list = r_list
        self.d_emb = d_emb

    def forward(self, v_list, mask):
        mid_states = []
        for enc_i, enc in enumerate(self.encoder_list):
            mid_states.append(enc(v_list[enc_i]).unsqueeze(1))  # B * D_v -> B * 1 * D_emb
        emb = torch.cat(mid_states, dim=1)  # B * V * D_emb
        emb = self.mid(emb)  # B * V * D_emb
        # emb = emb * mask.unsqueeze(2).expand(-1, -1, self.d_emb)

        rec_r = []
        for dec_i, dec in enumerate(self.decoder_list):
            rec_r.append(dec(emb))

        return emb, rec_r


class ModelSecond(nn.Module):
    def __init__(self, d_list, d_emb, n_enc_layer, n_dec_layer, n_cls, theta, dropout=0.1):
        super().__init__()
        r_list = []
        for i in range(len(d_list)):
            r_list.append(int(round(d_list[i] * theta)))

        self.ae = AutoEncoder(r_list, d_emb, n_enc_layer, n_dec_layer, dropout)
        self.classifier = Classifier(d_emb, n_cls, dropout)
        self.weights = nn.Parameter(torch.softmax(torch.zeros([1, len(d_list), 1]),dim=1))
        self.d_emb = d_emb

    def forward(self, v_list, mask_v):
        emb, rec_r = self.ae(v_list, mask_v)

        # weight fusion
        weight = torch.pow(self.weights.expand(emb.shape[0], -1, -1), 1)
        weight = torch.softmax(weight.masked_fill(mask_v.unsqueeze(2) == 0, -1e9), dim=1)
        emb_fus = torch.sum(emb * weight, dim=1)

        pred = self.classifier(emb_fus)

        return pred, rec_r
