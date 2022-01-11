from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def Swish(x):
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        input_dim = hidden_dim = latent_dim = args.edim
        eps = 1e-1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.act = Swish
        self.dropout = nn.Dropout(args.droprate)

    def forward(self, q_item_embs, user_hist):
        user_hist = user_hist.squeeze()
        hist_item_emb = q_item_embs(user_hist)
        hist_item_mask = (user_hist != 0).float()
        hist_item_mask = F.normalize(hist_item_mask, dim=-1)
        hist_item_mask = self.dropout(hist_item_mask)
        hist_item_emb *= hist_item_mask.unsqueeze(-1)
        x = hist_item_emb.sum(dim=1)

        h1 = self.ln1(self.act(self.fc1(x)))
        h2 = self.ln2(self.act(self.fc2(h1) + h1))
        h3 = self.ln3(self.act(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(self.act(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(self.act(self.fc5(h4) + h1 + h2 + h3 + h4))

        return self.fc_mu(h5), self.fc_logvar(h5)


class CompositePrior(nn.Module):
    def __init__(self, args):
        super(CompositePrior, self).__init__()
        latent_dim = args.edim
        self.mixture_weights = [0.15, 0.75, 0.1]

        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)

        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)

        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)

        self.encoder_old = Encoder(args)
        self.encoder_old.requires_grad_(False)

    def forward(self, q_item_embs, user_hist, z):
        self.encoder_old.eval()
        post_mu, post_logvar = self.encoder_old(q_item_embs, user_hist)

        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)

        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]

        density_per_gaussian = torch.stack(gaussians, dim=-1)

        return torch.logsumexp(density_per_gaussian, dim=-1)


class RecVAE(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(RecVAE, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.edim = edim = args.edim
        self.beta = args.beta

        self.q_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.p_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.q_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.p_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.encoder = Encoder(args)
        self.prior = CompositePrior(args)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
    def forward(self, batch):
        user, user_hist, user_nbrs, pos_item, neg_item = batch
        mu, logvar = self.encoder(self.q_item_embs, user_hist)
        hu = self.reparameterize(mu, logvar)

        pos_hi = self.p_item_embs(pos_item)
        neg_hi = self.p_item_embs(neg_item)
        pos_logits = (hu * pos_hi).sum(dim=-1)
        neg_logits = (hu * neg_hi).sum(dim=-1)

        if self.training:
            bce = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + \
                  F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
            prior = self.prior(self.q_item_embs, user_hist, hu)
            kld = self.beta * (log_norm_pdf(hu, mu, logvar) - prior).sum(dim=-1).mean()
            loss = bce + kld
            return loss
        else:
            return pos_logits, neg_logits

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))

    def eval_all_users(self, eval_loader):
        all_scores = list()
        self.eval()
        with torch.no_grad():
            for batch in eval_loader:
                user, user_hist, user_nbrs, eval_items = batch
                user_hist = user_hist.to(self.dev).long()
                eval_items = eval_items.to(self.dev).long()
                mu, logvar = self.encoder(self.q_item_embs, user_hist)
                hu = self.reparameterize(mu, logvar)
                hi = self.p_item_embs(eval_items)
                hu = hu.unsqueeze(1).expand_as(hi)
                batch_score = (hu * hi).sum(dim=-1)
                all_scores.append(batch_score)

            all_scores = torch.cat(all_scores, dim=0)
            return all_scores


class RecVAE_rate(nn.Module):
    def __init__(self, user_num, item_num, args):
        super(RecVAE_rate, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.dev = torch.device(args.device)
        self.edim = edim = args.edim
        self.beta = args.beta
        self.args = args

        self.q_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        self.p_item_embs = nn.Embedding(item_num, edim, padding_idx=0)
        nn.init.uniform_(self.q_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)
        nn.init.uniform_(self.p_item_embs.weight, a=-0.5 / item_num, b=0.5 / item_num)

        self.encoder = Encoder(args)
        self.prior = CompositePrior(args)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    # def forward(self, user_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=True):
    def forward(self, batch):
        user, u_ir, nbr, item, rate = batch
        user_hist = u_ir[:, :, 0]
        mu, logvar = self.encoder(self.q_item_embs, user_hist)
        hu = self.reparameterize(mu, logvar)
        hi = self.p_item_embs(item)
        logits = (hu * hi).sum(dim=-1)

        if self.training:

            if self.args.rescale_rate:
                pred_rate = torch.sigmoid(logits)
                true_rate = (rate - 1) / 4.0
            else:
                pred_rate = logits
                true_rate = rate

            mse = F.mse_loss(pred_rate, true_rate)
            prior = self.prior(self.q_item_embs, user_hist, hu)
            kld = self.beta * (log_norm_pdf(hu, mu, logvar) - prior).sum(dim=-1).mean()
            loss = mse + kld
            return loss
        else:
            return logits

    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))
