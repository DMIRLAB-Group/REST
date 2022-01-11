from argparse import ArgumentParser
from datetime import datetime
from time import time

import numpy as np
import torch.nn.functional as F

from models.REST import REST
from utils.data_loader import load_ds_causalrec_rate
from utils.utils import *


def parse_batch(batch, args):
    device = torch.device(f'{args.device}')
    user, u_ir, pos_nbr, neg_nbr, item, rate, unexp_item, unexp_rate = batch
    user = user.to(device).long()
    u_ir = u_ir.to(device).long()
    pos_nbr = pos_nbr.to(device).long()
    neg_nbr = neg_nbr.to(device).long()
    item = item.to(device).long()
    rate = rate.to(device).float()
    unexp_item = unexp_item.to(device).long()
    unexp_rate = unexp_rate.to(device).float()
    return user, u_ir, pos_nbr, neg_nbr, item, rate, unexp_item, unexp_rate


def train(model, opt, lr_shdr, loader, args):
    model.train()
    total_loss = 0.0
    train_temp = args.temp
    st = time()
    total_exp_label_loss = total_unexp_label_loss = total_nbr_loss = total_exp_loss = total_kld = total_emb_reg = 0.0

    for batch_idx, batch in enumerate(loader):
        user, u_ir, pos_nbr, neg_nbr, item, rate, unexp_item, unexp_rate = parse_batch(batch, args)
        opt.zero_grad()

        q_hidden, exposure_logits, exposure_label, \
        unexp_pred_rate, exp_pred_rate, \
        pos_nbr_logits, neg_nbr_logits, \
        user_emb, item_emb, unexp_item_emb = \
            model(user, u_ir, pos_nbr, neg_nbr, item, unexp_item, train_temp)

        unexp_indices = torch.where(unexp_rate != 0)[0]

        if args.rescale_rate:
            exp_pred_rate = torch.sigmoid(exp_pred_rate)
            rate = (rate - 1) / 4.0
            unexp_pred_rate = torch.sigmoid(unexp_pred_rate)
            unexp_rate = (unexp_rate - 1) / 4.0

        exp_label_loss = F.mse_loss(exp_pred_rate, rate)
        unexp_label_loss = F.mse_loss(unexp_pred_rate[unexp_indices], unexp_rate[unexp_indices])
        nbr_loss = F.binary_cross_entropy_with_logits(input=pos_nbr_logits, target=torch.ones_like(pos_nbr_logits)) + \
                   F.binary_cross_entropy_with_logits(input=neg_nbr_logits, target=torch.zeros_like(neg_nbr_logits))
        exp_loss = F.binary_cross_entropy_with_logits(input=exposure_logits, target=exposure_label)
        kld = get_kld(q_hidden, args)

        emb_reg = user_emb.norm(dim=-1).pow(2).mean() + item_emb.norm(dim=-1).pow(2).mean() + unexp_item_emb.norm(
            dim=-1).pow(2).mean()

        total_exp_label_loss += exp_label_loss.item()
        total_unexp_label_loss += unexp_label_loss.item()
        total_nbr_loss += nbr_loss.item()
        total_exp_loss += exp_loss.item()
        total_kld += kld.item()
        total_emb_reg += emb_reg.item()

        if batch_idx % 10 == 0 and False:
            val_metrics = evaluate(model, val_loader, args)
            val_mae, val_rmse = val_metrics
            logger.info('Val MAE={:.4f}, RMSE={:.4f}'.format(r, epoch, val_mae, val_rmse))

        loss = exp_label_loss + \
               args.w_unexp * unexp_label_loss + \
               args.alpha * nbr_loss + \
               exp_loss + \
               args.beta * kld + \
               args.emb_l2rg * emb_reg

        loss.backward()
        opt.step()
        lr_shdr.step()
        train_temp = np.maximum(args.temp * np.exp(-args.anneal_rate * batch_idx), args.temp_min)
        total_loss += loss.item()

    print('Train Loss={:.4f} Time={:.2f}s LR={:.6f}'.format(
        total_loss / len(loader), time() - st, lr_shdr.get_lr()))

    total_exp_label_loss /= len(loader)
    total_unexp_label_loss /= len(loader)
    total_nbr_loss /= len(loader)
    total_exp_loss /= len(loader)
    total_kld /= len(loader)
    total_emb_reg /= len(loader)

    print('EL={:.4f} unEL={:.4f} NL={:.4f} ExpL={:.4f} KL={:.4f} ER={:.4f}'.format(
        total_exp_label_loss, total_unexp_label_loss, total_nbr_loss, total_exp_loss, total_kld, total_emb_reg))


def evaluate(model, eval_loader, args):
    model.eval()
    total_mae = total_rmse = total = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            user, u_ir, pos_nbr, neg_nbr, item, rate, unexp_item, unexp_rate = parse_batch(batch, args)
            pred_rate = model.eval_forward(user, u_ir, pos_nbr, item)
            if args.rescale_rate:
                pred_rate = 1.0 + 4.0 * torch.sigmoid(pred_rate)

            diff = pred_rate - rate
            total_mae += diff.abs().sum().item()
            total_rmse += diff.pow(2).sum().item()
            total += len(diff)

        mae = total_mae / total
        rmse = np.sqrt(total_rmse / total)

        return mae, rmse


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Epinions')
    parser.add_argument('--model', default='REST')

    parser.add_argument('--edim', type=int, default=64)
    parser.add_argument('--seq_maxlen', type=int, default=50)
    parser.add_argument('--nbr_maxlen', type=int, default=50)
    parser.add_argument('--unexp_pos_threshold', type=int, default=3)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--temp_min', type=float, default=0.5)
    parser.add_argument('--anneal_rate', type=float, default=0.00001)
    parser.add_argument('--alpha', type=float, default=1.0, help='P(G|U) weight')
    parser.add_argument('--beta', type=float, default=30, help='KL weight')
    parser.add_argument('--w_unexp', type=float, default=0.01, help='unexp loss weight')
    parser.add_argument('--rescale_rate', type=int, default=0)

    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--droprate', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--l2rg', type=float, default=5e-4)
    parser.add_argument('--emb_l2rg', type=float, default=0.01)
    parser.add_argument('--lr_gamma', type=float, default=0.001)
    parser.add_argument('--lr_decayrate', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--check_epoch', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--categorical_dim', default=4, type=int)
    # latent_dim = edim // categorical_dim
    # parser.add_argument('--latent_dim', default=16, type=int)

    args = parser.parse_args()

    timestr = datetime.now().strftime('%Y%m%d_%Hh%Mm%Ss')
    model_path = f'../saved_models/{args.model}_{args.dataset}_best.pth'
    logger = get_logger(f'../logs/{args.model}_{args.dataset}_best.log')
    logger.info(args)
    device = torch.device(f'{args.device}')
    train_loader, val_loader, test_loader, user_num, item_num, train_u_ir = load_ds_causalrec_rate(args)
    metrics_list = []
    for r in range(1, args.repeat + 1):

        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = REST(user_num, item_num, args).to(device)
        # logger.info(str(model))
        opt = torch.optim.Adam(model.get_parameters(), lr=args.lr, weight_decay=args.l2rg)
        lr_shdr = StepwiseLR(opt, args.lr, gamma=args.lr_gamma, decay_rate=args.lr_decayrate)
        patience = 0.0
        best_mae = best_rmse = 1e6

        for epoch in range(1, args.max_epochs):
            train(model, opt, lr_shdr, train_loader, args)

            if epoch % args.check_epoch == 0:
                val_metrics = evaluate(model, val_loader, args)
                val_mae, val_rmse = val_metrics
                logger.info('Iter={} Epoch={} Val MAE={:.4f}, RMSE={:.4f}'.format(r, epoch, val_mae, val_rmse))

                if best_rmse > val_rmse:
                    print('Validation RMSE Increased {:.4f}-->{:.4f}'.format(best_rmse, val_rmse))
                    best_rmse = val_rmse
                    best_mae = val_mae
                    patience = 0
                    torch.save(model.state_dict(), model_path)
                else:
                    print('Patitence: {}/{}'.format(patience, args.patience))
                    patience += 1
                    if patience >= args.patience:
                        print('Early Stop !')
                        break

        # print('Saving overfitting model...')
        # torch.save(model.state_dict(), model_path)

        print('Testing...')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, args)
        metrics_list.append(test_metrics)
        logger.info('Iter={} Tst MAE={:.4f}, RMSE={:.4f}'.format(r, test_metrics[0], test_metrics[1]))
        logger.info('Iter={} Val MAE={:.4f}, RMSE={:.4f}'.format(r, best_mae, best_rmse))

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print('Test Summary:')
    logger.info('Mean MAE={:.4f}, RMSE={:.4f}'.format(means[0], means[1]))
    logger.info('Std  MAE={:.4f}, RMSE={:.4f}'.format(stds[0], stds[1]))
