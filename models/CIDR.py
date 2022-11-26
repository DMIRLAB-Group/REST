from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CIDR_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CIDR_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.user_confound_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_confound_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.user_bias = nn.Embedding(user_num, 1, padding_idx=0)
        self.item_bias = nn.Embedding(item_num, 1, padding_idx=0)
        self.alpha = 1.0
        self.beta = 1.0
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.user_confound_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_confound_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.user_bias.weight)
        self.glonal_bias = nn.Parameter(torch.zeros(()))


    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_emb = self.user_embs(user)
        item_emb = self.item_embs(item)
        user_confound_emb = self.user_confound_embs(user)
        item_confound_emb = self.item_confound_embs(item)
        user_b = self.user_bias(user).squeeze()
        item_b = self.item_bias(item).squeeze()

        logits = (user_emb * item_emb).sum(dim=-1) \
                 - self.alpha * (user_emb * item_confound_emb).sum(dim=-1) \
                 - self.beta * (user_confound_emb * item_emb).sum(dim=-1)\
                 + user_b + item_b + self.glonal_bias

        pred = torch.sigmoid(logits).squeeze()
        rate = (rate - 1) / 4.0
        mse_loss = F.mse_loss(pred, rate)

        emb_loss = self.args.emb_l2rg * (
                user_emb.pow(2).sum(dim=-1).mean() +
                item_emb.pow(2).sum(dim=-1).mean() +
                user_confound_emb.pow(2).sum(dim=-1).mean() +
                item_confound_emb.pow(2).sum(dim=-1).mean())

        loss = mse_loss + emb_loss
        return loss

    def eval_forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_emb = self.user_embs(user)
        item_emb = self.item_embs(item)
        user_confound_emb = self.user_confound_embs(user)
        item_confound_emb = self.item_confound_embs(item)
        user_b = self.user_bias(user).squeeze()
        item_b = self.item_bias(item).squeeze()

        logits = (user_emb * item_emb).sum(dim=-1) \
                 - self.alpha * (user_emb * item_confound_emb).sum(dim=-1) \
                 - self.beta * (user_confound_emb * item_emb).sum(dim=-1) \
                 + user_b + item_b + self.glonal_bias

        pred = torch.sigmoid(logits)

        return pred
