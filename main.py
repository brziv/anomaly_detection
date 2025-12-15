from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch import nn
import numpy as np
from utils import save_best_record
from timm.scheduler.cosine_lr import CosineLRScheduler
from torchinfo import summary
from tqdm import tqdm
import option
args=option.parse_args()
from model import Model
from dataset import Dataset
from train import train
from test import test
import datetime
import os
import random
import sys
import matplotlib.pyplot as plt


def save_config(save_path):
    path = save_path+'/'
    os.makedirs(path,exist_ok=True)
    f = open(path + "config_{}.txt".format(datetime.datetime.now()), 'w')
    for key in vars(args).keys():
        f.write('{}: {}'.format(key,vars(args)[key]))
        f.write('\n')

savepath = './ckpt/{}_{}_{}'.format(args.lr, args.batch_size, args.comment)
save_config(savepath)

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv1d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
         torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')     
    elif isinstance(m, nn.Conv3d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
if __name__ == '__main__':
    args=option.parse_args()
    random.seed(2025)
    np.random.seed(2025)
    torch.cuda.manual_seed(2025)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # DO NOT SHUFFLE, shuffling is handled by the Dataset class and not the DataLoader
    train_loader = DataLoader(Dataset(args, test_mode=False),
                               batch_size=args.batch_size // 2)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                             batch_size=args.batch_size)

    if args.model_arch == 'base':
        model = Model(dropout = args.dropout_rate, attn_dropout=args.attn_dropout_rate)
    elif args.model_arch == 'fast' or args.model_arch == 'tiny':
        model = Model(dropout = args.dropout_rate, attn_dropout=args.attn_dropout_rate, ff_mult = 1, dims = (32,32), depths = (1,1))
    else:
        print("Model architecture not recognized")
        sys.exit()
    model.apply(init_weights)

    if args.pretrained_ckpt is not None:
        model_ckpt = torch.load(args.pretrained_ckpt)
        model.load_state_dict(model_ckpt)
        print("pretrained loaded")

    model = model.to(device)

    if not os.path.exists('./ckpt'):
        os.makedirs('./ckpt')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay = 0.2)

    num_steps = len(train_loader)
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial= args.max_epoch * num_steps,
            cycle_mul=1.,
            lr_min=args.lr * 0.2,
            warmup_lr_init=args.lr * 0.01,
            warmup_t=args.warmup * num_steps,
            cycle_limit=20,
            t_in_epochs=False,
            warmup_prefix=True,
            cycle_decay = 0.95,
        )

    test_info = {"epoch": [], "test_AUC": [], "test_PR":[], "train_AUC": [], "train_PR": []}
    best_auc = -1
    best_model_path = None
    patience = 10
    best_epoch = 0

    for step in tqdm(
            range(0, args.max_epoch),
            total=args.max_epoch,
            dynamic_ncols=True
    ):

        cost, train_pr, train_auc = train(train_loader, model, optimizer, scheduler, device, step)
        scheduler.step(step + 1)

        auc, pr_auc = test(test_loader, model, args, device)

        test_info["epoch"].append(step)
        test_info["test_AUC"].append(auc)
        test_info["test_PR"].append(pr_auc)
        test_info["train_AUC"].append(train_auc)
        test_info["train_PR"].append(train_pr)
        
        if auc > best_auc:
            best_auc = auc
            best_epoch = step
            if best_model_path:
                os.remove(best_model_path)
            best_model_path = savepath + '/' + 'best.pkl'
            torch.save(model.state_dict(), best_model_path)
        elif step - best_epoch >= patience:
            print(f"Early stopping at epoch {step}, no improvement for {patience} epochs")
            break
        
        save_best_record(test_info, os.path.join(savepath, 'train_log.txt'))

    # Plot the metrics
    epochs = test_info["epoch"]
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, test_info["test_AUC"], label='Test AUC')
    plt.plot(epochs, test_info["train_AUC"], label='Train AUC')
    plt.title('AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(epochs, test_info["test_PR"], label='Test PR AUC')
    plt.plot(epochs, test_info["train_PR"], label='Train PR AUC')
    plt.title('PR AUC over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('PR AUC')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(epochs, test_info["test_AUC"], label='Test AUC')
    plt.title('Test AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, test_info["test_PR"], label='Test PR AUC')
    plt.title('Test PR AUC')
    plt.xlabel('Epoch')
    plt.ylabel('PR AUC')

    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'metrics_plot.png'))
    plt.close()
