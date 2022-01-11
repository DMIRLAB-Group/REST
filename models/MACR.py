import torch
import torch.nn as nn
import torch.nn.functional as F


class MACR_rank(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MACR_rank, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        self.alpha = args.alpha
        self.beta = args.beta
        self.c = args.c
        self.l2rg = args.l2rg

        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.user_lin = nn.Linear(args.edim, 1)
        self.item_lin = nn.Linear(args.edim, 1)

    def pred(self, hu, hi):
        pred_u = torch.sigmoid(self.user_lin(hu))
        pred_i = torch.sigmoid(self.item_lin(hi))
        pred_score = (hu * hi).sum(dim=-1, keepdims=True)
        return (pred_score - self.c) * pred_u * pred_i

    def forward(self, batch):
        uid, seq, nbr, pos, neg = batch
        neg_uid = torch.randint_like(uid, low=1, high=self.user_embs.weight.shape[0])
        user_emb = self.user_embs(uid)
        neg_user_emb = self.user_embs(neg_uid)
        pos_hi = self.item_embs(pos)
        neg_hi = self.item_embs(neg)

        pos_user_logits = self.user_lin(user_emb)
        neg_user_logits = self.user_lin(neg_user_emb)
        pos_item_logits = self.item_lin(pos_hi)
        neg_item_logits = self.item_lin(neg_hi)

        pos_user_prob = torch.sigmoid(pos_user_logits)
        # neg_user_prob = torch.sigmoid(neg_user_logits)
        pos_item_prob = torch.sigmoid(pos_item_logits)
        neg_item_prob = torch.sigmoid(neg_item_logits)

        pos_pair_logits = pos_user_prob * pos_item_prob * (user_emb * pos_hi).sum(-1)
        neg_pair_logits = pos_user_prob * neg_item_prob * (user_emb * neg_hi).sum(-1)

        pair_loss = F.binary_cross_entropy_with_logits(pos_pair_logits, torch.ones_like(pos_pair_logits)) + \
                    F.binary_cross_entropy_with_logits(neg_pair_logits, torch.zeros_like(neg_pair_logits))

        user_loss = F.binary_cross_entropy_with_logits(pos_user_logits, torch.ones_like(pos_user_logits)) + \
                    F.binary_cross_entropy_with_logits(neg_user_logits, torch.zeros_like(neg_user_logits))

        item_loss = F.binary_cross_entropy_with_logits(pos_item_logits, torch.ones_like(pos_item_logits)) + \
                    F.binary_cross_entropy_with_logits(neg_item_logits, torch.zeros_like(neg_item_logits))

        emb_loss = user_emb.pow(2).sum(dim=-1).mean() + \
                   pos_hi.pow(2).sum(dim=-1).mean() + \
                   neg_hi.pow(2).sum(dim=-1).mean()

        loss = pair_loss + self.alpha * user_loss + self.beta * item_loss + self.l2rg * emb_loss

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
                hi = self.item_embs(eval_items)
                hu = user_emb.unsqueeze(1).expand_as(hi)
                batch_scores = self.pred(hu, hi).squeeze()
                all_scores.append(batch_scores)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores


class MACR_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(MACR_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num

        self.alpha = args.alpha
        self.beta = args.beta
        self.c = args.c
        self.l2rg = args.l2rg
        self.rescale_rate = args.rescale_rate

        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.user_lin = nn.Linear(args.edim, 1)
        self.item_lin = nn.Linear(args.edim, 1)

    def pred(self, hu, hi):
        pred_u = torch.sigmoid(self.user_lin(hu))
        pred_i = torch.sigmoid(self.item_lin(hi))
        pred_score = (hu * hi).sum(dim=-1, keepdims=True)
        return (pred_score - self.c) * pred_u * pred_i

    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch

        neg_user = torch.randint_like(user, low=1, high=self.user_embs.weight.shape[0])
        neg_item = torch.randint_like(item, low=1, high=self.item_embs.weight.shape[0])

        user_emb = self.user_embs(user)
        neg_user_emb = self.user_embs(neg_user)
        pos_hi = self.item_embs(item)
        neg_hi = self.item_embs(neg_item)

        pos_user_logits = self.user_lin(user_emb)
        neg_user_logits = self.user_lin(neg_user_emb)
        pos_item_logits = self.item_lin(pos_hi)
        neg_item_logits = self.item_lin(neg_hi)

        pos_user_prob = torch.sigmoid(pos_user_logits)
        pos_item_prob = torch.sigmoid(pos_item_logits)

        pred_rate = pos_user_prob * pos_item_prob * (user_emb * pos_hi).sum(dim=-1, keepdims=True)
        if self.rescale_rate:
            pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)
        rate_loss = F.mse_loss(pred_rate.squeeze(), rate)

        user_loss = F.binary_cross_entropy_with_logits(pos_user_logits, torch.ones_like(pos_user_logits)) + \
                    F.binary_cross_entropy_with_logits(neg_user_logits, torch.zeros_like(neg_user_logits))

        item_loss = F.binary_cross_entropy_with_logits(pos_item_logits, torch.ones_like(pos_item_logits)) + \
                    F.binary_cross_entropy_with_logits(neg_item_logits, torch.zeros_like(neg_item_logits))

        reg_term = (torch.norm(user_emb) ** 2 + torch.norm(pos_hi) ** 2 + torch.norm(neg_hi) ** 2) / 2
        emb_loss = reg_term / user_emb.shape[0]

        loss = rate_loss + self.alpha * user_loss + self.beta * item_loss + self.l2rg * emb_loss

        return loss

    def eval_forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_emb = self.user_embs(user)
        item_emb = self.item_embs(item)
        pred_rate = self.pred(user_emb, item_emb)
        if self.rescale_rate:
            pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)
        return pred_rate.squeeze()
