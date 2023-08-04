import sys
import argparse
import os
import numpy as np
import torch
import torchvision
from utils import *


@torch.no_grad()
def validate_process(model, args):
    if args.dataset=='ggearth':
        import dataset as datasets
    else:
        import datasets_4cor_img as datasets
    model.eval()
    mace_list = []
    args.batch_size = 1
    val_dataset = datasets.fetch_dataloader(args, split='validation')
    for i_batch, data_blob in enumerate(val_dataset):
        image1, image2, image2w , flow_gt,  H  = [x.cuda() for x in data_blob]
        flow_gt = flow_gt.squeeze(0)
        flow_4cor = torch.zeros((2, 2, 2))
        flow_4cor[:, 0, 0] = flow_gt[:, 0, 0]
        flow_4cor[:, 0, 1] = flow_gt[:, 0, -1]
        flow_4cor[:, 1, 0] = flow_gt[:, -1, 0]
        flow_4cor[:, 1, 1] = flow_gt[:, -1, -1]

        image1 = image1.cuda()
        image2 = image2.cuda()

        four_pr = model(image1, image2, iters_lev0 = args.iters_lev0, iters_lev1 = args.iters_lev1, test_mode=True)
        mace = torch.sum((four_pr[0, :, :, :].cpu() - flow_4cor) ** 2, dim=0).sqrt()
        mace_list.append(mace.view(-1).numpy())
        torch.cuda.empty_cache()
        if i_batch>300:
            break

    model.train()
    mace = np.mean(np.concatenate(mace_list))
    print("Validation MACE: %f" % mace)
    return {'chairs_mace': mace}

