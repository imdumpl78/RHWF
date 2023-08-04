import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import torchgeometry as tgm
import os
import math
import random
from glob import glob
import os.path as osp
import cv2
from torchvision import datasets, transforms
from skimage import io
import torchvision

marginal = 32
patch_size = 128

composed_transform1 = transforms.Compose([transforms.ToPILImage(),transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25), transforms.ToTensor(),transforms.RandomErasing(p=0.25, scale=(0.01, 0.1), ratio=(0.3, 3.3), value='random')])

class homo_dataset(data.Dataset):
    def __init__(self):

        self.is_test = False
        self.init_seed = True
        self.image_list_img1 = []
        self.image_list_img2 = []
        self.dataset=[]
        self.colorjit = False

    def __getitem__(self, index):
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        img1 = cv2.imread(self.image_list_img1[index])
        img2 = cv2.imread(self.image_list_img2[index])

        if self.dataset=='mscoco':
            img1 = cv2.resize(img1, (320, 240))
            img2 = cv2.resize(img2, (320, 240))
        (height, width, _) = img1.shape

        y = random.randint(marginal, height - marginal - patch_size)
        x = random.randint(marginal, width - marginal - patch_size)

        # define corners of image patch
        top_left_point = (x, y)
        bottom_right_point = (patch_size + x, patch_size + y)

        perturbed_four_points_new = []

        top_left_point_new = (x, y)
        bottom_left_point_new = (x, patch_size + y - 1)
        bottom_right_point_new = (patch_size + x - 1, patch_size + y - 1)
        top_right_point_new = (x + patch_size - 1, y)
        four_points_new = [top_left_point_new, bottom_left_point_new, bottom_right_point_new, top_right_point_new]

        try:
            perturbed_four_points_new = []
            for i in range(4):
                      
                t1 = random.randint(-marginal, marginal)
                t2 = random.randint(-marginal, marginal)

                perturbed_four_points_new.append((four_points_new[i][0] + t1,
                                                  four_points_new[i][1] + t2))

            y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
            point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            org = np.float32(four_points_new)
            dst = np.float32(perturbed_four_points_new)
            H = cv2.getPerspectiveTransform(org, dst)
            H_inverse = np.linalg.inv(H)
        except:
            perturbed_four_points_new = []
            for i in range(4):
                t1 = 32//(i+1)
                t2 = -32//(i+1)
                perturbed_four_points_new.append((four_points_new[i][0] + t1,
                                                  four_points_new[i][1] + t2))

            y_grid, x_grid = np.mgrid[0:img1.shape[0], 0:img1.shape[1]]
            point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()

            org = np.float32(four_points_new)
            dst = np.float32(perturbed_four_points_new)
            H = cv2.getPerspectiveTransform(org, dst)
            H_inverse = np.linalg.inv(H)
        
        warped_image = cv2.warpPerspective(img2, H_inverse, (img1.shape[1], img1.shape[0]))

        img_patch_ori = img1[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0], :]
        img_patch_pert = warped_image[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0],:]

        warped_image2 = cv2.warpPerspective(img1, H_inverse, (img1.shape[1], img1.shape[0]))
        img_patch_pert_2 = warped_image2[top_left_point[1]:bottom_right_point[1], top_left_point[0]:bottom_right_point[0],:]

        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H).squeeze()

        diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        diff_x_branch1 = diff_x_branch1.reshape((img1.shape[0], img1.shape[1]))
        diff_y_branch1 = diff_y_branch1.reshape((img1.shape[0], img1.shape[1]))

        pf_patch_x_branch1 = diff_x_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch_y_branch1 = diff_y_branch1[top_left_point[1]:bottom_right_point[1],
                             top_left_point[0]:bottom_right_point[0]]

        pf_patch = np.zeros((patch_size, patch_size, 2))
        pf_patch[:, :, 0] = pf_patch_x_branch1
        pf_patch[:, :, 1] = pf_patch_y_branch1

        img_patch_ori = img_patch_ori[:, :, ::-1].copy()
        img_patch_pert = img_patch_pert[:, :, ::-1].copy()
        
        img1 = torch.from_numpy((img_patch_ori)).float().permute(2, 0, 1)
        img2 = torch.from_numpy((img_patch_pert)).float().permute(2, 0, 1)
        
        #################  downsample  #################
        #img1 = F.interpolate(F.interpolate(img1.unsqueeze(0), size=(128 // 4, 128 // 4), mode='bilinear'), size=(128, 128), mode='bilinear').squeeze(0)
        #img1 = F.interpolate(F.interpolate(img1.unsqueeze(0), size=(128 // 8, 128 // 8), mode='bilinear'), size=(128, 128), mode='bilinear').squeeze(0)
        #################  downsample  #################

        img1w = img_patch_pert_2[:, :, ::-1].copy()
        img1w = torch.from_numpy((img1w)).float().permute(2, 0, 1)
        
        if self.colorjit:
            img1 = composed_transform1(img1/255) * 255
            img2 = composed_transform1(img2/255) * 255
        flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()
        
        ### homo
        four_point_org = torch.zeros((2, 2, 2))
        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([128 - 1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, 128 - 1])
        four_point_org[:, 1, 1] = torch.Tensor([128 - 1, 128 - 1])

        four_point = torch.zeros((2, 2, 2))
        four_point[:, 0, 0] = flow[:, 0, 0] + torch.Tensor([0, 0])
        four_point[:, 0, 1] = flow[:, 0, -1] + torch.Tensor([128 - 1, 0])
        four_point[:, 1, 0] = flow[:, -1, 0] + torch.Tensor([0, 128 - 1])
        four_point[:, 1, 1] = flow[:, -1, -1] + torch.Tensor([128 - 1, 128 - 1])
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0)
        four_point = four_point.flatten(1).permute(1, 0).unsqueeze(0)
        H = tgm.get_perspective_transform(four_point_org, four_point)
        H = H.squeeze()

        return img2, img1, img1w, flow, H


class MYDATA(homo_dataset):
    def __init__(self, split='train', dataset='msococo'):
        super(MYDATA, self).__init__()
        if split == 'train':
            if dataset=='mscoco':
                root_img1 = '/UsrFile/yjc/xzq/ssddata/csy/mscoco2017/train2017'
                root_img2 = '/UsrFile/yjc/xzq/ssddata/csy/mscoco2017/train2017'
            if dataset=='ggmap':
                root_img1 = '/UsrFile/yjc/xzq/ssddata/csy/GoogleMap/train2014_input'
                root_img2 = '/UsrFile/yjc/xzq/ssddata/csy/GoogleMap/train2014_template_original'

        else:
            if dataset=='mscoco':
                root_img1 = '/UsrFile/yjc/xzq/ssddata/csy/mscoco2017/test2017'
                root_img2 = '/UsrFile/yjc/xzq/ssddata/csy/mscoco2017/test2017'
            if dataset=='ggmap':
                root_img1 = '/UsrFile/yjc/xzq/ssddata/csy/GoogleMap/val2014_input'
                root_img2 = '/UsrFile/yjc/xzq/ssddata/csy/GoogleMap/val2014_template_original'
            if dataset=='ggearth':
                root_img1 = '/UsrFile/yjc/xzq/ssddata/csy/GoogleEarth/val2014_input'
                root_img2 = '/UsrFile/yjc/xzq/ssddata/csy/GoogleEarth/val2014_template_original'

        self.colorjit = False
        self.dataset = dataset
        self.image_list_img1 = sorted(glob(osp.join(root_img1, '*.jpg')))
        self.image_list_img2 = sorted(glob(osp.join(root_img2, '*.jpg')))

    def __len__(self):
        return int(len(self.image_list_img1))


def fetch_dataloader(args, split='train'):

    if split == 'train':
        train_dataset = MYDATA(split='train',dataset=args.dataset)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True, num_workers=16, drop_last=False)
        print('Training with %d image pairs' % len(train_dataset))
    else: 
        train_dataset = MYDATA(split='val',dataset=args.dataset)
        train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size,
                                       pin_memory=True, shuffle=True, num_workers=16, drop_last=False)       
    
    return train_loader

