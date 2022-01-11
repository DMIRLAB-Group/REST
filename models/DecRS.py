import torch
import torch.nn as nn
import torch.nn.functional as F



class DecNFM_rate(nn.Module):
    def __init__(self, user_num, item_num, cate_num, cate_prior, args):
        super(DecNFM_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.cate_prior = torch.tensor(cate_prior, dtype=torch.float32).cuda().unsqueeze(dim=-1)

        self.dim = dim = args.edim
        self.l2rg = args.l2rg
        self.rescale_rate = args.rescale_rate

        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, dim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, dim, padding_idx=0)
        self.cate_embs = nn.Embedding(cate_num, dim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)
        nn.init.normal_(self.cate_embs.weight, std=0.01)

        self.dropout = nn.Dropout(args.droprate)
        self.dnn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(args.droprate),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(args.droprate),
            nn.Linear(dim, 1),
        )

    def decfm_forward(self, batch):
        user, u_ir, nbr, item, cate, rate = batch

        # user features
        user_emb = self.user_embs(user).view(-1, 1, self.dim)  # B x 1 x D

        # item features
        item_emb = self.item_embs(item).view(-1, 1, self.dim)  # B x 1 x D
        cate_emb = self.cate_embs(cate).view(-1, 1, self.dim)  # B x 1 x D

        # U & D => M: use FM to calculate M(\bar{d}, u)
        batch_size = user.shape[0]
        weighted_cate_embs = self.cate_embs.weight * self.cate_prior  # 1 x C x D
        weighted_cate_embs = weighted_cate_embs.expand((batch_size, self.cate_num, self.dim))
        user_confounder_emb = torch.cat([user_emb, weighted_cate_embs], dim=1)

        # confounder FM
        sum_sqr_user_confounder_emb = user_confounder_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_user_confounder_emb = (user_confounder_emb.pow(2)).sum(dim=1)  # B x D
        user_confounder_mediator = 0.5 * (sum_sqr_user_confounder_emb - sqr_sum_user_confounder_emb)  # B x D
        user_confounder_mediator = user_confounder_mediator.unsqueeze(dim=1)  # B x 1 x D

        # FM features
        fm_emb = torch.concat([
            user_emb,
            item_emb,
            cate_emb,
            user_confounder_mediator
        ], dim=1)  # B x 5 x D

        # FM feature interaction
        sum_sqr_fm_emb = fm_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_fm_emb = (fm_emb.pow(2)).sum(dim=1)  # B x D
        fm_out = 0.5 * (sum_sqr_fm_emb - sqr_sum_fm_emb)
        fm_out = self.dropout(fm_out)
        fm_logits = self.dnn(fm_out).squeeze()

        return fm_logits, user_emb, item_emb, cate_emb

    def forward(self, batch):
        fm_logits, user_emb, item_emb, cate_emb = self.decfm_forward(batch)
        pred_rate = fm_logits
        if self.rescale_rate:
            pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)
        rate = batch[-1].float()
        rate_loss = F.mse_loss(pred_rate, rate)
        reg_term = (torch.norm(user_emb) ** 2 + torch.norm(item_emb) ** 2 + torch.norm(cate_emb) ** 2) / 2
        emb_loss = reg_term / user_emb.shape[0]

        loss = rate_loss + self.l2rg * emb_loss
        # print('rl=%g el=%g' % (rate_loss.item(), emb_loss.item()))

        return loss

    def eval_forward(self, batch):
        pred_rate, _, _, _ = self.decfm_forward(batch)
        if self.rescale_rate:
            pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)
        return pred_rate


class DecFM_rate(nn.Module):
    def __init__(self, user_num, item_num, cate_num, cate_prior, args):
        super(DecFM_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.cate_num = cate_num
        self.cate_prior = torch.tensor(cate_prior, dtype=torch.float32).cuda().unsqueeze(dim=-1)

        self.dim = dim = args.edim
        self.l2rg = args.l2rg
        self.rescale_rate = args.rescale_rate

        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, dim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, dim, padding_idx=0)
        self.cate_embs = nn.Embedding(cate_num, dim, padding_idx=0)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)
        nn.init.normal_(self.cate_embs.weight, std=0.01)

        self.dropout = nn.Dropout(args.droprate)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

    # def decfm_forward_v0(self, batch):
    #     user, u_ir, nbr, item, cate, rate = batch
    #
    #     # user features
    #     item_seq = u_ir[:, :, 0]
    #     item_seq_emb = self.item_embs(item_seq)
    #     item_seq_len = torch.sign(item_seq).sum(dim=1, keepdim=True)
    #     user_hist_emb = item_seq_emb.sum(dim=1) / (item_seq_len + 1e-9)
    #     user_hist_emb = user_hist_emb.view(-1, 1, self.dim)  # B x 1 x D
    #     user_emb = self.user_embs(user).view(-1, 1, self.dim)  # B x 1 x D
    #
    #     # item features
    #     item_emb = self.item_embs(item).view(-1, 1, self.dim)  # B x 1 x D
    #     cate_emb = self.cate_embs(cate).view(-1, 1, self.dim)  # B x 1 x D
    #
    #     # U & D => M: use FM to calculate M(\bar{d}, u)
    #     batch_size = user.shape[0]
    #     weighted_cate_embs = self.cate_embs.weight * self.cate_prior  # 1 x C x D
    #     weighted_cate_embs = weighted_cate_embs.expand((batch_size, self.cate_num, self.dim))
    #     user_confounder_emb = torch.cat([user_emb, user_hist_emb, weighted_cate_embs], dim=1)
    #
    #     # confounder FM
    #     sum_sqr_user_confounder_emb = user_confounder_emb.sum(dim=1).pow(2)  # B x D
    #     sqr_sum_user_confounder_emb = (user_confounder_emb.pow(2)).sum(dim=1)  # B x D
    #     user_confounder_mediator = 0.5 * (sum_sqr_user_confounder_emb - sqr_sum_user_confounder_emb)  # B x D
    #     user_confounder_mediator = user_confounder_mediator.unsqueeze(dim=1)  # B x 1 x D
    #
    #     # FM features
    #     fm_emb = torch.concat([
    #         user_emb,
    #         user_hist_emb,
    #         item_emb,
    #         cate_emb,
    #         user_confounder_mediator
    #     ], dim=1)  # B x 5 x D
    #
    #     # FM feature interaction
    #     sum_sqr_fm_emb = fm_emb.sum(dim=1).pow(2)  # B x D
    #     sqr_sum_fm_emb = (fm_emb.pow(2)).sum(dim=1)  # B x D
    #     fm_out = 0.5 * (sum_sqr_fm_emb - sqr_sum_fm_emb)
    #     fm_logits = self.dropout(fm_out).sum(dim=-1) + self.bias_
    #
    #     return fm_logits, user_emb, item_emb, cate_emb

    def decfm_forward(self, batch):
        user, u_ir, nbr, item, cate, rate = batch

        # user features
        user_emb = self.user_embs(user).view(-1, 1, self.dim)  # B x 1 x D

        # item features
        item_emb = self.item_embs(item).view(-1, 1, self.dim)  # B x 1 x D
        cate_emb = self.cate_embs(cate).view(-1, 1, self.dim)  # B x 1 x D

        # U & D => M: use FM to calculate M(\bar{d}, u)
        batch_size = user.shape[0]
        weighted_cate_embs = self.cate_embs.weight * self.cate_prior  # 1 x C x D
        weighted_cate_embs = weighted_cate_embs.expand((batch_size, self.cate_num, self.dim))
        user_confounder_emb = torch.cat([user_emb, weighted_cate_embs], dim=1)

        # confounder FM
        sum_sqr_user_confounder_emb = user_confounder_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_user_confounder_emb = (user_confounder_emb.pow(2)).sum(dim=1)  # B x D
        user_confounder_mediator = 0.5 * (sum_sqr_user_confounder_emb - sqr_sum_user_confounder_emb)  # B x D
        user_confounder_mediator = user_confounder_mediator.unsqueeze(dim=1)  # B x 1 x D

        # FM features
        fm_emb = torch.concat([
            user_emb,
            item_emb,
            cate_emb,
            user_confounder_mediator
        ], dim=1)  # B x 5 x D

        # FM feature interaction
        sum_sqr_fm_emb = fm_emb.sum(dim=1).pow(2)  # B x D
        sqr_sum_fm_emb = (fm_emb.pow(2)).sum(dim=1)  # B x D
        fm_out = 0.5 * (sum_sqr_fm_emb - sqr_sum_fm_emb)
        fm_logits = self.dropout(fm_out).sum(dim=-1) + self.bias_

        return fm_logits, user_emb, item_emb, cate_emb

    def forward(self, batch):
        fm_logits, user_emb, item_emb, cate_emb = self.decfm_forward(batch)
        pred_rate = fm_logits
        if self.rescale_rate:
            pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)
        rate = batch[-1].float()
        rate_loss = F.mse_loss(pred_rate, rate)
        reg_term = (torch.norm(user_emb) ** 2 + torch.norm(item_emb) ** 2 + torch.norm(cate_emb) ** 2) / 2
        emb_loss = reg_term / user_emb.shape[0]

        loss = rate_loss + self.l2rg * emb_loss
        # print('rl=%g el=%g' % (rate_loss.item(), emb_loss.item()))

        return loss

    def eval_forward(self, batch):
        pred_rate, _, _, _ = self.decfm_forward(batch)
        if self.rescale_rate:
            pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)
        return pred_rate
