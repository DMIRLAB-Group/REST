from argparse import ArgumentParser
from datetime import datetime

import numpy as np
import torch
from models.DecRS import DecNFM_rate

from utils.dec_data_loader import load_ds_recfm_rate
from utils.logger import get_logger


def parse_batch(batch):
    user, u_ir, nbr, item, cate, rate = batch
    return [x.cuda().long() for x in batch]


def train(model, opt, loader, args):
    model.train()
    total_loss = 0.0

    from time import time
    st = time()

    for batch in loader:
        parsed_batch = parse_batch(batch)
        opt.zero_grad()
        loss = model(parsed_batch)
        loss.backward()
        opt.step()
        total_loss += loss.item()

    # total_time = float(time() - st)
    # avg_batch_time = total_time / len(loader)
    # print("Batch num :", len(loader))
    # print("Avg 1-batch time: %g" % avg_batch_time)
    # print("Train loss: %g" % (total_loss / len(loader)))


def evaluate(model, eval_loader, args):
    model.eval()
    total_mae = total_mse = total = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            parsed_batch = parse_batch(batch)
            pred_rate = model.eval_forward(parsed_batch)
            true_rate = parsed_batch[-1].float()
            diff = pred_rate - true_rate
            total_mae += diff.abs().sum().item()
            total_mse += diff.pow(2).sum().item()
            total += len(diff)

        mae = total_mae / total
        rmse = np.sqrt(total_mse / total)

        return mae, rmse


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='Ciao')
    parser.add_argument('--model', default='DecNFM')
    parser.add_argument("--edim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=1024*4)
    parser.add_argument('--droprate', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--l2rg', type=float, default=5e-5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--seed', type=int, default=27)
    parser.add_argument('--check_epoch', type=int, default=1)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--rescale_rate', type=int, default=0)
    args = parser.parse_args()

    timestr = datetime.now().strftime("%Y%m%d_%Hh%Mm%Ss")
    model_path = f'../saved_models/{args.model}_{args.dataset}_{timestr}.pth'
    logger = get_logger(f'../logs/{args.model}_{args.dataset}_{timestr}')
    logger.info(args)
    device = torch.device(f'{args.device}')

    train_loader, val_loader, test_loader, user_num, item_num, cate_num, cate_prior = \
        load_ds_recfm_rate(
            args=args,
            dataset=args.dataset,
            batch_size=args.batch_size,
            test_size=0.2,
            split='fo',
            seed=args.seed,
            item_maxlen=20,
            nbrs_maxlen=20,
            num_worker=args.num_workers)

    print(f'user={user_num} '
          f'item={item_num} '
          f'cate={cate_num} '
          f'train={len(train_loader)} '
          f'val={len(val_loader)} '
          f'test={len(test_loader)}')

    metrics_list = []
    for r in range(1, args.repeat + 1):

        seed = args.seed + r
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        model = DecNFM_rate(user_num, item_num, cate_num, cate_prior, args).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2rg)
        patience = 0
        best_mae = best_rmse = 1e6

        for epoch in range(1, args.max_epochs):
            train(model, opt, train_loader, args)

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

        print('Testing...')
        model.load_state_dict(torch.load(model_path))
        test_metrics = evaluate(model, test_loader, args)
        metrics_list.append(test_metrics)
        logger.info('Iter={} Tst MAE={:.4f}, RMSE={:.4f}'.format(r, test_metrics[0], test_metrics[1]))
        logger.info('Iter={} Val MAE={:.4f}, RMSE={:.4f}'.format(r, best_mae, best_rmse))
        logger.info('-------------------------------------------------------------------------------')
        del model

    metrics = np.array(metrics_list)
    means = metrics.mean(axis=0)
    stds = metrics.std(axis=0)
    print('Test Summary:')
    logger.info('Mean MAE={:.4f}, RMSE={:.4f}'.format(means[0], means[1]))
    logger.info('Std  MAE={:.4f}, RMSE={:.4f}'.format(stds[0], stds[1]))
