import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
    def __init__(self,
                 edim,
                 n_layers,
                 user_num,
                 item_num,
                 ui_graph,
                 args):
        super(LightGCN, self).__init__()
        self.n_layers = n_layers
        self.num_users = user_num
        self.num_items = item_num
        self.args = args
        self.dev = torch.device(args.device)
        self.Graph = ui_graph.to(self.dev)

        self.embedding_user = torch.nn.Embedding(self.num_users, edim)
        self.embedding_item = torch.nn.Embedding(self.num_items, edim)
        nn.init.uniform_(self.embedding_user.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [torch.cat([users_emb, items_emb])]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)

        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, batch):
        uid, pos, neg = batch
        indices = torch.where(pos != 0)
        uid = uid.unsqueeze(1).expand_as(pos)

        users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0 = \
            self.getEmbedding(
                uid.long().to(self.dev),
                pos.long().to(self.dev),
                neg.long().to(self.dev))

        pos_scores = (users_emb * pos_emb).sum(dim=-1)
        neg_scores = (users_emb * neg_emb).sum(dim=-1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores[indices] - pos_scores[indices]))
        loss += self.args.emb_reg * 0.5 * (userEmb0.norm() + posEmb0.norm() + negEmb0.norm())

        return loss


class LightGCN_rate(nn.Module):
    def __init__(self,
                 n_layers,
                 user_num,
                 item_num,
                 ui_graph,
                 args):
        super(LightGCN_rate, self).__init__()
        edim = args.edim
        self.n_layers = n_layers
        self.num_users = user_num
        self.num_items = item_num
        self.args = args
        self.dev = torch.device(args.device)
        self.Graph = ui_graph.to(self.dev)

        self.embedding_user = torch.nn.Embedding(self.num_users, edim)
        self.embedding_item = torch.nn.Embedding(self.num_items, edim)
        nn.init.uniform_(self.embedding_user.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item.weight, a=-0.5 / item_num, b=0.5 / item_num)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [torch.cat([users_emb, items_emb])]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(self.Graph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def getEmbedding(self, users, items):
        all_users, all_items = self.computer()

        users_emb = all_users[users]
        items_emb = all_items[items]

        users_emb_ego = self.embedding_user(users)
        items_emb_ego = self.embedding_item(items)

        return users_emb, items_emb, users_emb_ego, items_emb_ego

    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        users_emb, items_emb, users_emb_ego, items_emb_ego = self.getEmbedding(user, item)
        pred_rate = (users_emb * items_emb).sum(dim=-1)

        return pred_rate, users_emb_ego, items_emb_ego

    def eval_forward(self, all_users, all_items, batch):
        user, u_ir, nbr, item, rate = batch
        users_emb = all_users[user]
        items_emb = all_items[item]
        pred_rate = (users_emb * items_emb).sum(dim=-1)

        return pred_rate
