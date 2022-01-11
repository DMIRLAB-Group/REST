import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GraphRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)

        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.user_encoder = UserEncoder(args)
        self.rate_predictor = RatePredictor(args)

    def forward(self, batch):
        user, user_hist, user_nbrs, pos_item, neg_item = batch
        hu = self.user_encoder(self.user_embs, self.item_embs, user, user_hist, user_nbrs)
        pos_hi = self.item_embs(pos_item)
        neg_hi = self.item_embs(neg_item)
        pos_logits, neg_logits = self.rate_predictor(hu, pos_hi, neg_hi)

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for batch in eval_loader:
                user, user_hist, user_nbrs, eval_items = batch
                user = user.to(self.dev).long()
                user_hist = user_hist.to(self.dev).long()
                user_nbrs = user_nbrs.to(self.dev).long()
                eval_items = eval_items.to(self.dev).long()
                h_user = self.user_encoder(self.user_embs, self.item_embs, user, user_hist, user_nbrs)
                h_item = self.item_embs(eval_items)
                batch_score = self.rate_predictor.eval_foward(h_user, h_item)
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores

    def get_parameters(self):
        param_list = [
            {'params': self.user_encoder.parameters()},
            {'params': self.rate_predictor.parameters()},
            {'params': self.user_embs.parameters(), 'weight_decay': 0.0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0.0}
        ]

        return param_list


class GraphRec_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(GraphRec_rate, self).__init__()
        rate_num = 5 + 1
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)

        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.rate_embs = nn.Embedding(rate_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.rate_embs.weight, a=-0.5 / rate_num, b=0.5 / rate_num)

        self.user_encoder = UserEncoder_rate(args)
        self.rate_predictor = RatePredictor(args)

    def forward(self, batch):
        user, user_hist, user_nbrs, item, rate = batch
        hu = self.user_encoder(self.user_embs, self.item_embs, self.rate_embs, user, user_hist, user_nbrs)
        hi = self.item_embs(item)
        pred_rate = self.rate_predictor.eval_foward(hu, hi)

        return pred_rate

    def get_parameters(self):
        param_list = [
            {'params': self.user_encoder.parameters()},
            {'params': self.rate_predictor.parameters()},
            {'params': self.user_embs.parameters(), 'weight_decay': 0.0},
            {'params': self.item_embs.parameters(), 'weight_decay': 0.0}
        ]

        return param_list


class RatePredictor(nn.Module):
    def __init__(self, args):
        super(RatePredictor, self).__init__()
        edim = args.edim
        droprate = args.droprate

        self.user_lin1 = nn.Linear(edim, edim)
        # self.user_bn = nn.BatchNorm1d(edim, momentum=args.momentum)
        self.user_lin2 = nn.Linear(edim, edim)

        self.item_lin1 = nn.Linear(edim, edim)
        # self.item_bn = nn.BatchNorm1d(edim, momentum=args.momentum)
        self.item_lin2 = nn.Linear(edim, edim)

        self.rate_predictor = nn.Sequential(
            nn.Linear(2 * edim, edim),
            # nn.BatchNorm1d(user_dim, momentum=args.momentum),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(edim, edim),
            # nn.BatchNorm1d(user_dim, momentum=args.momentum),
            nn.ReLU(),
            nn.Dropout(droprate),
            nn.Linear(edim, 1))

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, hu, pos_hi, neg_hi):
        hu = self.user_lin2(self.dropout(self.act(self.user_lin1(hu))))
        pos_hi = self.item_lin2(self.dropout(self.act(self.item_lin1(pos_hi))))
        neg_hi = self.item_lin2(self.dropout(self.act(self.item_lin1(neg_hi))))
        pos_h = torch.cat([hu, pos_hi], dim=-1)
        neg_h = torch.cat([hu, neg_hi], dim=-1)
        pos_logits = self.rate_predictor(pos_h)
        neg_logits = self.rate_predictor(neg_h)

        return pos_logits, neg_logits

    def eval_foward(self, hu, hi):
        hu = self.user_lin2(self.dropout(self.act(self.user_lin1(hu))))
        hi = self.item_lin2(self.dropout(self.act(self.item_lin1(hi))))
        if len(hu.shape) != len(hi.shape):
            hu = hu.unsqueeze(1).expand_as(hi)
        logits = self.rate_predictor(torch.cat([hu, hi], dim=-1))
        return logits.squeeze()


class UserEncoder(nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        edim = args.edim
        droprate = args.droprate
        self.item_attn = AttnLayer(edim, droprate)
        self.user_attn = AttnLayer(edim, droprate)
        self.fuse_lin = nn.Linear(edim + edim, edim)
        self.self_lin = nn.Linear(edim + edim, edim)
        self.act = nn.ReLU()

    def forward(self, user_embs, item_embs, user, user_hist, user_nbrs):
        user_emb = user_embs(user)

        # Aggregate user history items
        user_hist = user_hist.squeeze()
        hist_item_emb = item_embs(user_hist)
        hist_item_mask = -1e8 * (user_hist == 0).long()
        h_item = self.item_attn(hist_item_emb, user_emb, hist_item_mask)  # B x d

        # Aggregate user neighbors
        nbrs_emb = user_embs(user_nbrs)  # B x l x d
        nbrs_mask = -1e8 * (user_nbrs == 0).long()
        h_social = self.user_attn(nbrs_emb, user_emb, nbrs_mask)

        # Concat two components
        h = torch.cat([h_item, h_social], dim=-1)
        h = self.act(self.fuse_lin(h))

        # Concat self embedding
        h = torch.cat([h, user_emb], dim=-1)
        h = self.self_lin(h)

        return h


class UserEncoder_rate(nn.Module):
    def __init__(self, args):
        super(UserEncoder_rate, self).__init__()
        edim = args.edim
        droprate = args.droprate
        self.item_attn = AttnLayer(edim, droprate)
        self.user_attn = AttnLayer(edim, droprate)
        self.rate_lin = nn.Linear(edim + edim, edim)
        self.fuse_lin = nn.Linear(edim + edim, edim)
        self.self_lin = nn.Linear(edim + edim, edim)
        self.act = nn.ReLU()

    def forward(self, user_embs, item_embs, rate_embs, user, user_hist, user_nbrs):
        user_emb = user_embs(user)

        # Aggregate user history items
        user_hist_item = user_hist[:, :, 0]
        user_hist_rate = user_hist[:, :, 1]
        hist_item_emb = item_embs(user_hist_item)
        hist_rate_emb = rate_embs(user_hist_rate)
        hist_item_emb = self.rate_lin(torch.cat([hist_item_emb, hist_rate_emb], dim=-1))
        hist_item_mask = -1e8 * (user_hist_item == 0).long()
        h_item = self.item_attn(hist_item_emb, user_emb, hist_item_mask)  # B x d

        # Aggregate user neighbors
        nbrs_emb = user_embs(user_nbrs)  # B x l x d
        nbrs_mask = -1e8 * (user_nbrs == 0).long()
        h_social = self.user_attn(nbrs_emb, user_emb, nbrs_mask)

        # Concat two components
        h = torch.cat([h_item, h_social], dim=-1)
        h = self.act(self.fuse_lin(h))

        # Concat self embedding
        h = torch.cat([h, user_emb], dim=-1)
        h = self.self_lin(h)

        return h


class ItemEncoder(nn.Module):
    def __init__(self):
        super(ItemEncoder, self).__init__()

    def forward(self, item_embs_mu, item):
        item_emb = item_embs_mu(item)
        return item_emb


class AttnLayer(nn.Module):
    def __init__(self, edim, droprate):
        super(AttnLayer, self).__init__()
        self.attn1 = nn.Linear(2 * edim, edim)
        self.attn2 = nn.Linear(edim, 1)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, items, user, items_mask):
        user = user.unsqueeze(1).expand_as(items)
        h = torch.cat([items, user], dim=-1)
        h = self.act(self.attn1(h))
        h = self.attn2(h) + items_mask.unsqueeze(-1).float()
        a = F.softmax(h, dim=1)
        attn_out = (a * items).sum(dim=1)
        return attn_out
