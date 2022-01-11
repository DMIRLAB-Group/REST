from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausE_rank(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CausE_rank, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
        self.cf_distance = args.cf_distance
        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.user_bias = nn.Embedding(user_num, 1, padding_idx=0)
        self.item_bias = nn.Embedding(item_num, 1, padding_idx=0)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_bias.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.user_bias.weight, a=-0.5 / user_num, b=0.5 / user_num)

        # self.glonal_bias = nn.Parameter(torch.zeros(()))
        self.alpha = nn.Parameter(1e-6 * torch.ones(()))

    def forward(self, batch):
        uid, seq, nbr, pos, neg = batch

        user_emb = self.user_embs(uid)
        pos_item_emb = self.item_embs(pos)
        neg_item_emb = self.item_embs(neg)

        user_b = self.user_bias(uid).squeeze()
        pos_item_b = self.item_bias(pos).squeeze()
        neg_item_b = self.item_bias(neg).squeeze()

        pos_logits = self.alpha * (user_emb * pos_item_emb).sum(dim=-1)  # + user_b + pos_item_b
        neg_logits = self.alpha * (user_emb * neg_item_emb).sum(dim=-1)  # + user_b + neg_item_b

        bce_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + \
                   F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

        emb_loss = self.args.l2rg * (
                user_emb.pow(2).sum(dim=-1).mean() +
                pos_item_emb.pow(2).sum(dim=-1).mean() +
                neg_item_emb.pow(2).sum(dim=-1).mean()
            # user_b.pow(2).sum(dim=-1).mean() +
            # pos_item_b.pow(2).sum(dim=-1).mean() +
            # pos_item_b.pow(2).sum(dim=-1).mean()
        )

        factual_loss = bce_loss + emb_loss

        counter_factual_loss = 0
        conctrol_emb = self.item_embs.weight[0]
        normed_conctrol_emb = F.normalize(conctrol_emb, dim=-1)
        normed_pos_item_emb = F.normalize(pos_item_emb, dim=-1)
        normed_neg_item_emb = F.normalize(neg_item_emb, dim=-1)
        normed_conctrol_emb = normed_conctrol_emb.unsqueeze(0).expand_as(normed_pos_item_emb)

        if self.cf_distance == 'l2':
            counter_factual_loss = F.mse_loss(normed_conctrol_emb, normed_pos_item_emb) + \
                                   F.mse_loss(normed_conctrol_emb, normed_neg_item_emb)

        elif self.cf_distance == 'l1':
            counter_factual_loss = F.l1_loss(normed_conctrol_emb, normed_pos_item_emb) + \
                                   F.l1_loss(normed_conctrol_emb, normed_neg_item_emb)

        elif self.cf_distance == 'cos':
            counter_factual_loss = F.cosine_similarity(normed_conctrol_emb, normed_pos_item_emb, dim=-1).mean() + \
                                   F.cosine_similarity(normed_conctrol_emb, normed_neg_item_emb, dim=-1).mean()

        loss = factual_loss + self.args.w_cf * counter_factual_loss

        return loss

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for batch in eval_loader:
                uid, seq, nbr, eval_items = batch
                uid = uid.to(self.dev).long()
                eval_items = eval_items.to(self.dev).long()

                user_emb = self.user_embs(uid)
                item_emb = self.item_embs(eval_items)
                user_emb = user_emb.unsqueeze(1).expand_as(item_emb)
                # user_b = self.user_bias(uid).squeeze()
                # item_b = self.item_bias(eval_items).squeeze()
                # user_b = user_b.unsqueeze(1).expand_as(item_b)

                batch_scores = self.alpha * (user_emb * item_emb).sum(dim=-1)  # + user_b + item_b
                all_scores.append(batch_scores)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores


class CausE_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(CausE_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.args = args
        self.cf_distance = args.cf_distance
        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.user_bias = nn.Embedding(user_num, 1, padding_idx=0)
        self.item_bias = nn.Embedding(item_num, 1, padding_idx=0)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_bias.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.user_bias.weight, a=-0.5 / user_num, b=0.5 / user_num)

        self.glonal_bias = nn.Parameter(torch.zeros(()))
        self.alpha = nn.Parameter(1e-6 * torch.ones(()))

    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_emb = self.user_embs(user)
        item_emb = self.item_embs(item)
        user_b = self.user_bias(user).squeeze()
        item_b = self.item_bias(item).squeeze()

        logits = self.alpha * (user_emb * item_emb).sum(dim=-1) + \
                 user_b + item_b + self.glonal_bias

        pred = torch.sigmoid(logits).squeeze()

        # print('logits=', logits.shape, logits)
        # print('pred=', pred.shape, pred)

        # log_loss = F.binary_cross_entropy_with_logits(logits, rate)
        rate = (rate - 1) / 4.0
        mse_loss = F.mse_loss(pred, rate)

        # print('mse=', mse_loss)

        emb_loss = self.args.emb_l2rg * (
                user_emb.pow(2).sum(dim=-1).mean() + item_emb.pow(2).sum(dim=-1).mean() +
                user_b.pow(2).sum(dim=-1).mean() + item_b.pow(2).sum(dim=-1).mean())

        factual_loss = mse_loss + emb_loss

        counter_factual_loss = 0
        conctrol_emb = self.item_embs.weight[0]
        normed_conctrol_emb = F.normalize(conctrol_emb, dim=-1)
        normed_item_emb = F.normalize(item_emb, dim=-1)
        normed_conctrol_emb = normed_conctrol_emb.unsqueeze(0).expand_as(normed_item_emb)

        # print(normed_item_emb.shape)
        # print(normed_conctrol_emb.shape)

        if self.cf_distance == 'l2':
            counter_factual_loss = F.mse_loss(normed_conctrol_emb, normed_item_emb)
        elif self.cf_distance == 'l1':
            counter_factual_loss = F.l1_loss(normed_conctrol_emb, normed_item_emb)
        elif self.cf_distance == 'cos':
            counter_factual_loss = F.cosine_similarity(normed_conctrol_emb, normed_item_emb, dim=-1).mean()

        loss = factual_loss + self.args.w_cf * counter_factual_loss

        # print('f=', factual_loss)
        # print('cf=', counter_factual_loss)
        # exit()

        return loss

    def eval_forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_emb = self.user_embs(user)
        item_emb = self.item_embs(item)
        user_b = self.user_bias(user).squeeze()
        item_b = self.item_bias(item).squeeze()

        logits = self.alpha * (user_emb * item_emb).sum(dim=-1) + \
                 user_b + item_b + self.glonal_bias

        pred = torch.sigmoid(logits)

        return pred
