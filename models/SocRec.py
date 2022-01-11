import torch
import torch.nn as nn


class SocRec(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SocRec, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        uid, seq, nbr, pos, neg = batch

        user_emb = self.user_embs(uid)
        pos_hi = self.item_embs(pos)
        neg_hi = self.item_embs(neg)
        hu = user_emb
        pos_logits = self.pred(hu, pos_hi)
        neg_logits = self.pred(hu, neg_hi)

        nbrs_emb = self.user_embs(nbr)
        hu = user_emb.unsqueeze(1).expand_as(nbrs_emb)
        nbr_logits = self.pred(hu, nbrs_emb)

        return pos_logits, neg_logits, nbr_logits, user_emb, nbrs_emb, pos_hi, neg_hi

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
                batch_scores = self.pred(hu, hi)
                all_scores.append(batch_scores)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores


class SocRec_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SocRec_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.user_embs = nn.Embedding(user_num, args.edim, padding_idx=0)
        self.item_embs = nn.Embedding(item_num, args.edim, padding_idx=0)
        nn.init.uniform_(self.user_embs.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def pred(self, hu, hi):
        return (hu * hi).sum(dim=-1)

    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch

        user_emb = self.user_embs(user)

        item_emb = self.item_embs(item)
        pos_logits = self.pred(user_emb, item_emb)

        nbr_indices = torch.where(nbr != 0)
        nbrs_emb = self.user_embs(nbr)
        hu = user_emb.unsqueeze(1).expand_as(nbrs_emb)
        nbr_logits = self.pred(hu, nbrs_emb)
        nbr_logits = nbr_logits[nbr_indices]

        return pos_logits, nbr_logits, user_emb, nbrs_emb, item_emb
