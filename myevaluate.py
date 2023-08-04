import sys
import numpy as np
import random
import cv2
import os
import torch
import torchvision
import argparse
import scipy

import datasets_4cor_img as datasets
from rhwf import RHWF
from utils import *


def evaluate_SNet(model, val_dataset, batch_size=0, args = None):

    assert batch_size > 0, "batchsize > 0"

    total_mace = torch.empty(0)

    for i_batch, data_blob in enumerate(val_dataset):
        img1, img2, image2w, flow_gt,  H  = [x.cuda() for x in data_blob]

        if i_batch==0:
            if not os.path.exists('watch'):
                os.makedirs('watch')
            save_img(torchvision.utils.make_grid((img1)), './watch/' + 'test_img1.bmp')
            save_img(torchvision.utils.make_grid((img2)), './watch/' + 'test_img2.bmp')

        four_pred = model(img1, img2, iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1, test_mode=True)

        flow_4cor = torch.zeros((four_pred.shape[0], 2, 2, 2))
        flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
        flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
        flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
        flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]
    
        mace_ = (flow_4cor - four_pred.cpu().detach())**2
        mace_ = ((mace_[:,0,:,:] + mace_[:,1,:,:])**0.5)
        mace_vec = torch.mean(torch.mean(mace_, dim=1), dim=1)
      
        total_mace = torch.cat([total_mace,mace_vec], dim=0)
        final_mace = torch.mean(total_mace).item()
        print(mace_.mean())
        print("MACE Metric: ", final_mace)
        sys.exit(0)
    if not os.path.exists("res_mat"):
        os.makedirs("res_mat")
    scipy.io.savemat('res_mat/' + args.savemat, {'matrix': total_mace.numpy()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/mscoco_6/120000_RHWF.pth',help="restore checkpoint")
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')
    parser.add_argument('--savemat', type=str,  default='test.mat')
    
    parser.add_argument('--lev0', default=True, action='store_true', help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true', help='warp once')
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=0)

    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    
    args = parser.parse_args()
    device = torch.device('cuda:'+ str(args.gpuid[0]))
    setup_seed(2022)
    
    model = RHWF(args)
    model_med = torch.load(args.model)['net']
    model.load_state_dict(model_med)

    model.to(device) 
    model.eval()

    batchsz = 1
    args.batch_size = batchsz
    if args.dataset=='ggearth':
        import dataset as datasets
    val_dataset = datasets.fetch_dataloader(args, split='validation')

    evaluate_SNet(model, val_dataset, batch_size=batchsz, args=args)
    
