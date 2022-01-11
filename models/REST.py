import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class REST(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(REST, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.categorical_dim = args.categorical_dim
        self.latent_dim = int(args.edim // args.categorical_dim)
        self.dev = torch.device(args.device)

        self.user_embs_mu = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs_mu = nn.Embedding(item_num, args.edim, padding_idx=0)
        self.rate_embs = nn.Embedding(5 + 1, args.edim, padding_idx=0)
        nn.init.uniform_(self.item_embs_mu.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.user_embs_mu.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.rate_embs.weight, a=-0.5 / 6, b=0.5 / 6)

        # User encoder P(u | g)
        self.user_encoder = UserEncoder_rate(args)

        # Rec Strategy Inference block Q(s | g, u, e, i, a)
        self.strategy_encoder = StrategyEncoder(args, latent_dim=self.latent_dim, categorical_dim=self.categorical_dim)

        # Exposure predictor P(e | u, s, i)
        self.exposure_predictor = ExposurePredictor(args)

        # Rate predictor P(a | u, i, e)
        self.rate_predictor = RatePredictor(args)

    def generate(self, user, user_hist, user_nbrs, s_sample, unexp_item, exp_item, unexp_strategy, exp_strategy):
        # User Representations (tesnor)
        h_unexp_user, h_exp_user = self.user_encoder(self.user_embs_mu, self.item_embs_mu, self.rate_embs, user,
                                                     user_hist, user_nbrs)

        # Item Embeddings (tensor)
        h_unexp_item = self.item_embs_mu(unexp_item)
        h_exp_item = self.item_embs_mu(exp_item)
        h_item = torch.cat([h_unexp_item, h_exp_item], dim=0)
        h_user = torch.cat([h_unexp_user, h_exp_user], dim=0)

        # Exposure indicator (Ditribution)
        exposure_logits = self.exposure_predictor(h_user, h_item, s_sample)

        # Rate predictor (tensor)
        unexp_rate = self.rate_predictor(h_unexp_user, h_unexp_item, unexp_strategy, exposed=False)
        exp_rate = self.rate_predictor(h_exp_user, h_exp_item, exp_strategy, exposed=True)

        return exposure_logits, unexp_rate, exp_rate

    def forward(self, user, user_hist, pos_nbr, neg_nbr, item, unexp_item, temp):
        # Inference
        stategy_sample, q_hidden, unexp_strategy, exp_strategy = \
            self.strategy_encoder(self.user_embs_mu, self.item_embs_mu, self.rate_embs,
                                  user, user_hist, pos_nbr, unexp_item, item, temp)

        # Generate
        exposure_logits, unexp_pred_rate, exp_pred_rate = \
            self.generate(user, user_hist, pos_nbr, stategy_sample, unexp_item, item, unexp_strategy, exp_strategy)

        # delete padding items
        all_items = torch.cat([unexp_item, item], dim=-1)
        exp_indices = torch.where(all_items != 0)[0]
        exposure_label = torch.cat([torch.zeros_like(unexp_item), torch.ones_like(item)], dim=0).float()
        exposure_logits = exposure_logits[exp_indices]
        exposure_label = exposure_label[exp_indices]

        # P(g | u)
        nbr_indices = torch.where(pos_nbr != 0)[0]
        user_emb = self.user_embs_mu(user)
        pos_nbrs_emb = self.user_embs_mu(pos_nbr)
        neg_nbrs_emb = self.user_embs_mu(neg_nbr)
        user_emb = user_emb.unsqueeze(1).expand_as(pos_nbrs_emb)
        pos_nbr_logits = (user_emb * pos_nbrs_emb).sum(dim=-1)[nbr_indices]
        neg_nbr_logits = (user_emb * neg_nbrs_emb).sum(dim=-1)[nbr_indices]

        # Return emb for reg
        user_emb = self.user_embs_mu(user)
        item_emb = self.item_embs_mu(item)
        unexp_item_emb = self.item_embs_mu(unexp_item)

        return q_hidden, \
               exposure_logits, exposure_label, \
               unexp_pred_rate, exp_pred_rate, \
               pos_nbr_logits, neg_nbr_logits, \
               user_emb, item_emb, unexp_item_emb

    def eval_forward(self, user, user_hist, user_nbrs, item):
        # Inference
        exp_strategy = self.strategy_encoder.eval_forward(
            self.user_embs_mu, self.item_embs_mu, self.rate_embs,
            user, user_hist, user_nbrs, item)

        _, exp_h_user = self.user_encoder(
            self.user_embs_mu, self.item_embs_mu, self.rate_embs, user, user_hist, user_nbrs)
        h_item = self.item_embs_mu(item)
        pred_rate = self.rate_predictor(exp_h_user, h_item, exp_strategy, exposed=True)
        return pred_rate

    def get_parameters(self):
        param_list = [
            {'params': self.user_encoder.parameters()},
            {'params': self.strategy_encoder.parameters()},
            {'params': self.exposure_predictor.parameters()},
            {'params': self.rate_predictor.parameters()},
            {'params': self.user_embs_mu.parameters(), 'weight_decay': 0.0},
            {'params': self.item_embs_mu.parameters(), 'weight_decay': 0.0},
            {'params': self.rate_embs.parameters(), 'weight_decay': 0.0}
        ]

        return param_list


class ExposurePredictor(nn.Module):
    def __init__(self, args):
        super(ExposurePredictor, self).__init__()
        edim = args.edim
        droprate = args.droprate
        self.mlp = nn.Sequential(nn.Linear(edim + edim + edim, edim),
                                 nn.ReLU(),
                                 nn.Dropout(droprate),
                                 nn.Linear(edim, 1))

    def forward(self, h_user, h_item, h_strategy):
        logits = self.mlp(torch.cat([h_user, h_item, h_strategy], dim=-1))
        return logits.squeeze()


class RatePredictor(nn.Module):
    def __init__(self, args):
        super(RatePredictor, self).__init__()
        edim = args.edim
        droprate = args.droprate

        self.user_mlp = nn.Sequential(
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(edim, edim),
            nn.ReLU(),
            nn.Linear(edim, edim),
        )

        self.unexposed_predictor = nn.Sequential(
            nn.Linear(3 * edim, edim),
            nn.ReLU(),
            nn.Linear(edim, edim),
            nn.ReLU(),
            nn.Linear(edim, 1))

        self.exposed_predictor = nn.Sequential(
            nn.Linear(3 * edim, edim),
            nn.ReLU(),
            nn.Linear(edim, edim),
            nn.ReLU(),
            nn.Linear(edim, 1))

    def forward(self, hu, hi, hs, exposed):
        h = torch.cat([self.user_mlp(hu), self.item_mlp(hi), hs], dim=-1)
        logits = self.exposed_predictor(h) if exposed else self.unexposed_predictor(h)
        return logits.squeeze()


class StrategyEncoder(nn.Module):
    def __init__(self, args, latent_dim=8, categorical_dim=8):
        super(StrategyEncoder, self).__init__()
        user_dim = item_dim = s_dim = args.edim
        droprate = args.droprate
        self.edim = args.edim
        self.latent_dim = latent_dim
        self.categorical_dim = categorical_dim

        self.s_user_encoder = UserEncoder_rate(args)

        self.s_item_encoder = nn.Sequential(nn.Linear(item_dim, item_dim),
                                            nn.ReLU(),
                                            nn.Dropout(droprate),
                                            nn.Linear(item_dim, item_dim))

        self.unexp_merge_mlp = nn.Sequential(nn.Linear(user_dim + item_dim, s_dim),
                                             nn.ReLU())

        self.exp_merge_mlp = nn.Sequential(nn.Linear(user_dim + item_dim, s_dim),
                                           nn.ReLU())

    def forward(self, user_embs, item_embs, rate_embs, user, user_hist, user_nbrs, unexp_item, exp_item, temp):
        unexp_h_user, exp_h_user = self.s_user_encoder(user_embs, item_embs, rate_embs, user, user_hist, user_nbrs)
        unexp_h_item = self.s_item_encoder(item_embs(unexp_item))
        exp_h_item = self.s_item_encoder(item_embs(exp_item))
        unexp_h = self.unexp_merge_mlp(torch.cat([unexp_h_user, unexp_h_item], dim=-1))
        exp_h = self.exp_merge_mlp(torch.cat([exp_h_user, exp_h_item], dim=-1))
        q = torch.cat([unexp_h, exp_h], dim=0)
        q_y = q.view(-1, self.latent_dim, self.categorical_dim)
        strategy_sample = gumbel_softmax(q_y, temp, self.latent_dim, self.categorical_dim)
        return strategy_sample, F.softmax(q, dim=-1), unexp_h, exp_h

    def eval_forward(self, user_embs, item_embs, rate_embs, user, user_hist, user_nbrs, exp_item):
        unexp_h_user, exp_h_user = self.s_user_encoder(user_embs, item_embs, rate_embs, user, user_hist, user_nbrs)
        exp_h_item = self.s_item_encoder(item_embs(exp_item))
        exp_h = self.exp_merge_mlp(torch.cat([exp_h_user, exp_h_item], dim=-1))
        return exp_h


def gumbel_softmax_sample(logits, temperature, eps=1e-20):
    U = torch.rand_like(logits)
    gumbel_sample = -Variable(torch.log(-torch.log(U + eps) + eps))
    y = logits + gumbel_sample
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature, latent_dim, categorical_dim):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard.view(-1, latent_dim * categorical_dim)


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
        self.exp_lin = nn.Linear(edim, edim)
        self.unexp_lin = nn.Linear(edim, edim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(droprate)

    def forward(self, user_embs, item_embs, rate_embs, user, user_hist, user_nbrs):
        user_emb = user_embs(user)

        # Aggregate user history items
        user_hist_item = user_hist[:, :, 0]
        user_hist_rate = user_hist[:, :, 1]
        hist_item_emb = item_embs(user_hist_item)
        hist_rate_emb = rate_embs(user_hist_rate)

        hist_item_emb = self.rate_lin(torch.cat([hist_item_emb, hist_rate_emb], dim=-1))
        hist_item_mask = -1e9 * (user_hist_item == 0).long()
        h_item = self.item_attn(hist_item_emb, user_emb, hist_item_mask)  # B x d

        # Aggregate user neighbors
        nbrs_emb = user_embs(user_nbrs)  # B x l x d
        nbrs_mask = -1e9 * (user_nbrs == 0).long()
        h_social = self.user_attn(nbrs_emb, user_emb, nbrs_mask)

        # Concat two components
        h = torch.cat([h_item, h_social], dim=-1)
        h = self.dropout(self.act(self.fuse_lin(h)))

        h = torch.cat([h, user_emb], dim=-1)
        h = self.self_lin(h)

        unexp_h = self.unexp_lin(h)
        exp_h = self.exp_lin(h)

        return unexp_h, exp_h


class ItemEncoder(nn.Module):
    def __init__(self):
        super(ItemEncoder, self).__init__()

    def forward(self, item_embs_mu, item):
        item_emb = item_embs_mu(item)
        return item_emb


class AttnLayer(nn.Module):
    def __init__(self, edim, droprate):
        super(AttnLayer, self).__init__()
        self.attn1 = nn.Linear(edim + edim, edim)
        self.attn2 = nn.Linear(edim, edim)
        self.attn3 = nn.Linear(edim, 1)
        self.act = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(droprate)

    def forward(self, items_emb, user_emb, items_mask):
        user_emb = user_emb.unsqueeze(1).expand_as(items_emb)
        h = torch.cat([items_emb, user_emb], dim=-1)
        h = self.dropout(self.act(self.attn1(h)))
        h = self.dropout(self.act(self.attn2(h)))
        h = self.attn3(h) + items_mask.unsqueeze(-1).float()
        a = F.softmax(h, dim=1)
        attn_out = (a * items_emb).sum(dim=1)
        return attn_out

# class AttnLayer(nn.Module):
#     def __init__(self, edim, droprate):
#         super(AttnLayer, self).__init__()
#
#     def forward(self, items_emb, user_emb, items_mask):
#         user_emb = user_emb.unsqueeze(1).expand_as(items_emb)
#         h = (items_emb * user_emb).sum(dim=-1, keepdims=True) + items_mask.unsqueeze(-1).float()
#         a = F.softmax(h, dim=1)
#         attn_out = (a * items_emb).sum(dim=1)
#         return attn_out, a
