import torch
import torch.nn as nn


class NeuMF(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(NeuMF, self).__init__()
        edim = args.edim
        self.dev = torch.device(args.device)
        self.embedding_user_mlp = torch.nn.Embedding(user_num, edim, padding_idx=0)
        self.embedding_item_mlp = torch.nn.Embedding(item_num, edim, padding_idx=0)
        self.embedding_user_mf = torch.nn.Embedding(user_num, edim, padding_idx=0)
        self.embedding_item_mf = torch.nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.embedding_user_mlp.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item_mlp.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.embedding_user_mf.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item_mf.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.mlp_lin0 = nn.Linear(edim + edim, edim)
        self.mlp_lin1 = nn.Linear(edim, edim)
        self.mlp_lin2 = nn.Linear(edim, edim)
        self.out_lin = nn.Linear(edim + edim, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, batch):
        uid, seq, nbr, pos, neg = batch
        user_indices = uid
        item_indices = pos
        neg_item_indices = neg

        pos_item_embedding_mlp = self.embedding_item_mlp(item_indices)
        neg_item_embedding_mlp = self.embedding_item_mlp(neg_item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        pos_item_embedding_mf = self.embedding_item_mf(item_indices)
        neg_item_embedding_mf = self.embedding_item_mf(neg_item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)

        pos_mlp_vector = torch.cat([user_embedding_mlp, pos_item_embedding_mlp], dim=-1)
        pos_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        pos_mlp_vector)))))))))

        pos_mf_vector = self.dropout(torch.mul(user_embedding_mf, pos_item_embedding_mf))
        pos_vector = torch.cat([pos_mlp_vector, pos_mf_vector], dim=-1)
        pos_logits = self.out_lin(pos_vector).squeeze()

        neg_mlp_vector = torch.cat([user_embedding_mlp, neg_item_embedding_mlp], dim=-1)
        neg_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        neg_mlp_vector)))))))))

        neg_mf_vector = self.dropout(torch.mul(user_embedding_mf, neg_item_embedding_mf))
        neg_vector = torch.cat([neg_mlp_vector, neg_mf_vector], dim=-1)
        neg_logits = self.out_lin(neg_vector).squeeze()

        return pos_logits, neg_logits

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for batch in eval_loader:
                uid, seq, nbr, eval_items = batch
                user_indices = uid.long().to(self.dev)
                item_indices = eval_items.long().to(self.dev)

                pos_item_embedding_mlp = self.embedding_item_mlp(item_indices)
                user_embedding_mlp = self.embedding_user_mlp(user_indices).unsqueeze(1).expand_as(
                    pos_item_embedding_mlp)
                pos_item_embedding_mf = self.embedding_item_mf(item_indices)
                user_embedding_mf = self.embedding_user_mf(user_indices).unsqueeze(1).expand_as(pos_item_embedding_mf)

                pos_mlp_vector = torch.cat([user_embedding_mlp, pos_item_embedding_mlp], dim=-1)
                pos_mlp_vector = \
                    self.dropout(self.act(self.mlp_lin2(
                        self.dropout(self.act(self.mlp_lin1(
                            self.dropout(self.act(self.mlp_lin0(
                                pos_mlp_vector)))))))))

                pos_mf_vector = self.dropout(torch.mul(user_embedding_mf, pos_item_embedding_mf))
                pos_vector = torch.cat([pos_mlp_vector, pos_mf_vector], dim=-1)
                batch_scores = self.out_lin(pos_vector).squeeze()

                all_scores.append(batch_scores)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores


class NeuMF_rate(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 args):
        super(NeuMF_rate, self).__init__()
        edim = args.edim
        self.dev = torch.device(args.device)
        self.embedding_user_mlp = torch.nn.Embedding(user_num, edim, padding_idx=0)
        self.embedding_item_mlp = torch.nn.Embedding(item_num, edim, padding_idx=0)
        self.embedding_user_mf = torch.nn.Embedding(user_num, edim, padding_idx=0)
        self.embedding_item_mf = torch.nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.embedding_user_mlp.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item_mlp.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.embedding_user_mf.weight, a=-0.5 / user_num, b=0.5 / user_num)
        nn.init.uniform_(self.embedding_item_mf.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.mlp_lin0 = nn.Linear(edim + edim, edim)
        self.mlp_lin1 = nn.Linear(edim, edim)
        self.mlp_lin2 = nn.Linear(edim, edim)
        self.out_lin = nn.Linear(edim + edim, 1)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch

        user_indices = user
        item_indices = item

        pos_item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        pos_item_embedding_mf = self.embedding_item_mf(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)

        pos_mlp_vector = torch.cat([user_embedding_mlp, pos_item_embedding_mlp], dim=-1)
        pos_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        pos_mlp_vector)))))))))

        pos_mf_vector = self.dropout(torch.mul(user_embedding_mf, pos_item_embedding_mf))
        pos_vector = torch.cat([pos_mlp_vector, pos_mf_vector], dim=-1)
        pos_logits = self.out_lin(pos_vector).squeeze()

        return pos_logits

    def cvib_forward(self, batch):
        user, u_ir, nbr, item, rate = batch

        user_indices = user
        item_indices = item
        neg_item_indices = torch.randint_like(item, low=1, high=self.embedding_item_mf.weight.shape[0])

        pos_item_embedding_mlp = self.embedding_item_mlp(item_indices)
        neg_item_embedding_mlp = self.embedding_item_mlp(neg_item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        pos_item_embedding_mf = self.embedding_item_mf(item_indices)
        neg_item_embedding_mf = self.embedding_item_mf(neg_item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)

        pos_mlp_vector = torch.cat([user_embedding_mlp, pos_item_embedding_mlp], dim=-1)
        pos_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        pos_mlp_vector)))))))))

        pos_mf_vector = self.dropout(torch.mul(user_embedding_mf, pos_item_embedding_mf))
        pos_vector = torch.cat([pos_mlp_vector, pos_mf_vector], dim=-1)
        pos_logits = self.out_lin(pos_vector).squeeze()

        neg_mlp_vector = torch.cat([user_embedding_mlp, neg_item_embedding_mlp], dim=-1)
        neg_mlp_vector = \
            self.dropout(self.act(self.mlp_lin2(
                self.dropout(self.act(self.mlp_lin1(
                    self.dropout(self.act(self.mlp_lin0(
                        neg_mlp_vector)))))))))

        neg_mf_vector = self.dropout(torch.mul(user_embedding_mf, neg_item_embedding_mf))
        neg_vector = torch.cat([neg_mlp_vector, neg_mf_vector], dim=-1)
        neg_logits = self.out_lin(neg_vector).squeeze()

        return pos_logits, neg_logits
