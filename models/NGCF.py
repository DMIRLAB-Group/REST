import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF_rate(nn.Module):
    def __init__(self,
                 n_layers,
                 user_num,
                 item_num,
                 ui_graph,
                 args):
        super(NGCF_rate, self).__init__()
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

        self.lins = nn.ModuleDict()
        for layer in range(n_layers):
            self.lins.add_module(name=f'lin_{layer}_gc', module=nn.Linear(edim, edim))
            self.lins.add_module(name=f'lin_{layer}_bi', module=nn.Linear(edim, edim))

        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(args.droprate)

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ego_emb = torch.cat([users_emb, items_emb])
        embs = [ego_emb]
        for layer in range(self.n_layers):
            side_emb = torch.sparse.mm(self.Graph, ego_emb)
            sum_emb = self.lins[f'lin_{layer}_gc'](side_emb)
            bi_emb = self.lins[f'lin_{layer}_bi'](ego_emb * side_emb)
            ego_emb = self.dropout(self.act(sum_emb + bi_emb))
            norm_emb = F.normalize(ego_emb, p=2, dim=-1)
            embs.append(norm_emb)

        embs = torch.stack(embs, dim=1)
        ngcf_out = torch.mean(embs, dim=1)
        users, items = torch.split(ngcf_out, [self.num_users, self.num_items])
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
