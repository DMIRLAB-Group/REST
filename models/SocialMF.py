import torch
import torch.nn as nn


class SocialMF(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SocialMF, self).__init__()
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

        batch_size, nbr_maxlen = nbr.shape
        nbr = nbr.cpu()
        nbr_mask = torch.BoolTensor(nbr == 0).to(self.dev)
        nbr = nbr.to(self.dev)
        nbr_len = (nbr_maxlen - nbr_mask.sum(1)).view(batch_size, 1)
        nbr_len = torch.maximum(nbr_len, torch.ones_like(nbr_len))
        nbr_emb = self.user_embs(nbr)
        nbr_emb *= ~nbr_mask.unsqueeze(-1)
        nbr_emb = nbr_emb.sum(dim=1) / nbr_len

        return pos_logits, neg_logits, user_emb, nbr_emb, pos_hi, neg_hi

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


class SocialMF_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(SocialMF_rate, self).__init__()
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

        # Get mask and neighbors length
        batch_size, nbr_maxlen = nbr.shape
        nbr_mask = torch.BoolTensor(nbr.cpu() == 0).to(self.dev)  # B x nl
        nbr_len = (nbr_maxlen - nbr_mask.sum(1))  # B
        nbr_emb = self.user_embs(nbr)  # B x nl x d
        nbr_emb *= ~nbr_mask.unsqueeze(-1)  # B x nl x d
        nbr_len = nbr_len.view(batch_size, 1)  # B x 1
        nbr_emb = nbr_emb.sum(dim=1) / nbr_len  # B x d
        return pos_logits, user_emb, nbr_emb, item_emb
