import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from skimage import io
import random
import sys


def sequence_loss(four_preds, flow_gt, H, gamma, args):
    """ Loss function defined over sequence of flow predictions """

    flow_4cor = torch.zeros((four_preds[0].shape[0], 2, 2, 2)).to(four_preds[0].device)
    flow_4cor[:,:, 0, 0]  = flow_gt[:,:, 0, 0]
    flow_4cor[:,:, 0, 1] = flow_gt[:,:,  0, -1]
    flow_4cor[:,:, 1, 0] = flow_gt[:,:, -1, 0]
    flow_4cor[:,:, 1, 1] = flow_gt[:,:, -1, -1]

    ce_loss = 0.0

    for i in range(args.iters_lev0):
        i_weight = gamma**(args.iters_lev0 - i - 1)
        i4cor_loss = (four_preds[i] - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()

    for i in range(args.iters_lev0, args.iters_lev1 + args.iters_lev0):
        i_weight = gamma ** (args.iters_lev1 + args.iters_lev0 - i - 1)
        i4cor_loss = (four_preds[i] - flow_4cor).abs()
        ce_loss += i_weight * (i4cor_loss).mean()

    mace = torch.sum((four_preds[-1] - flow_4cor)**2, dim=1).sqrt()
    metrics = {
        '1px': (mace < 1).float().mean().item(),
        '3px': (mace < 3).float().mean().item(),
        'mace': mace.mean().item(),
    }
    return ce_loss , metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    return optimizer, scheduler


class Logger_(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass


class Logger:
    def __init__(self, model, scheduler, args):
        self.model = model
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_mace_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(np.int)
        time_left_hms = "{:02d}h{:02d}m{:02d}s".format(time_left_sec // 3600, time_left_sec % 3600 // 60, time_left_sec % 3600 % 60)
        time_left_hms = f"{time_left_hms:>12}"
        # print the training status
        print(training_str + metrics_str + time_left_hms)

        # logging running loss to total loss
        self.train_mace_list.append(np.mean(self.running_loss_dict['mace']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}


def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_mace_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()
