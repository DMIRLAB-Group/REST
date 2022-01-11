import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiVAE(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MultiVAE, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.edim = edim = args.edim

        self.q_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.p_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.q_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.p_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.q_lin1 = nn.Linear(edim, edim)
        self.q_lin2 = nn.Linear(edim, edim + edim)

        self.p_lin1 = nn.Linear(edim, edim)
        self.p_lin2 = nn.Linear(edim, edim)

        self.act = nn.Tanh()
        self.dropout = nn.Dropout(args.droprate)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def enc(self, user_hist):
        user_hist = user_hist.squeeze()
        hist_item_emb = self.q_item_embs(user_hist)
        hist_item_mask = (user_hist != 0).float()
        hist_item_mask = F.normalize(hist_item_mask, dim=-1)
        hist_item_mask = self.dropout(hist_item_mask)
        hist_item_emb *= hist_item_mask.unsqueeze(-1)
        hu = hist_item_emb.sum(dim=1)
        mu, logvar = self.q_lin2(self.act(self.q_lin1(hu))).chunk(2, dim=-1)
        hu = self.reparameterize(mu, logvar)
        hu = self.p_lin2(self.act(self.p_lin1(hu)))
        return hu, mu, logvar

    def forward(self, batch):
        user, user_hist, user_nbrs, pos_item, neg_item = batch
        hu, mu, logvar = self.enc(user_hist)
        pos_hi = self.p_item_embs(pos_item)
        neg_hi = self.p_item_embs(neg_item)
        pos_logits = (hu * pos_hi).sum(dim=-1)
        neg_logits = (hu * neg_hi).sum(dim=-1)
        return pos_logits, neg_logits, mu, logvar

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for batch in eval_loader:
                user, user_hist, user_nbrs, eval_items = batch
                user_hist = user_hist.to(self.dev).long()
                eval_items = eval_items.to(self.dev).long()
                hu, mu, logvar = self.enc(user_hist)
                hi = self.p_item_embs(eval_items)
                hu = hu.unsqueeze(1).expand_as(hi)
                batch_score = (hu * hi).sum(dim=-1)
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores


class MultiVAE_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MultiVAE_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.edim = edim = args.edim

        self.q_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.p_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.q_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.p_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.q_lin1 = nn.Linear(edim, edim)
        self.q_lin2 = nn.Linear(edim, edim + edim)

        self.p_lin1 = nn.Linear(edim, edim)
        self.p_lin2 = nn.Linear(edim, edim)

        self.act = nn.Tanh()
        self.dropout = nn.Dropout(args.droprate)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def enc(self, user_hist):
        user_hist = user_hist.squeeze()
        hist_item_emb = self.q_item_embs(user_hist)
        hist_item_mask = (user_hist != 0).float()
        hist_item_mask = F.normalize(hist_item_mask, dim=-1)
        hist_item_mask = self.dropout(hist_item_mask)
        hist_item_emb *= hist_item_mask.unsqueeze(-1)
        hu = hist_item_emb.sum(dim=1)
        mu, logvar = self.q_lin2(self.act(self.q_lin1(hu))).chunk(2, dim=-1)
        hu = self.reparameterize(mu, logvar)
        hu = self.p_lin2(self.act(self.p_lin1(hu)))
        return hu, mu, logvar

    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_hist = u_ir[:, :, 0]
        hu, mu, logvar = self.enc(user_hist)
        hi = self.p_item_embs(item)
        logits = (hu * hi).sum(dim=-1)
        return logits, mu, logvar
