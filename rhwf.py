import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchgeometry as tgm
import scipy.io as io

from utils import *
from encoder import RHWF_Encoder
from decoder import GMA_update
from ATT.attention_layer import Correlation, FocusFormer_Attention


class Get_Flow(nn.Module):
    def __init__(self, sz):
        super().__init__()
        self.sz = sz

    def forward(self, four_point, a):
        four_point = four_point/ torch.Tensor([a]).cuda()

        four_point_org = torch.zeros((2, 2, 2)).cuda()

        four_point_org[:, 0, 0] = torch.Tensor([0, 0])
        four_point_org[:, 0, 1] = torch.Tensor([self.sz[3]-1, 0])
        four_point_org[:, 1, 0] = torch.Tensor([0, self.sz[2]-1])
        four_point_org[:, 1, 1] = torch.Tensor([self.sz[3]-1, self.sz[2]-1])

        four_point_org = four_point_org.unsqueeze(0)
        four_point_org = four_point_org.repeat(self.sz[0], 1, 1, 1)

        four_point_new = four_point_org + four_point

        four_point_org = four_point_org.flatten(2).permute(0, 2, 1)
        four_point_new = four_point_new.flatten(2).permute(0, 2, 1)
        H = tgm.get_perspective_transform(four_point_org, four_point_new)
        gridy, gridx = torch.meshgrid(torch.linspace(0, self.sz[3]-1, steps=self.sz[3]), torch.linspace(0, self.sz[2]-1, steps=self.sz[2]))
        points = torch.cat((gridx.flatten().unsqueeze(0), gridy.flatten().unsqueeze(0), torch.ones((1, self.sz[3] * self.sz[2]))),
                           dim=0).unsqueeze(0).repeat(self.sz[0], 1, 1).to(four_point.device)
        points_new = H.bmm(points)
        points_new = points_new / points_new[:, 2, :].unsqueeze(1)
        points_new = points_new[:, 0:2, :]
        flow = torch.cat((points_new[:, 0, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1),
                          points_new[:, 1, :].reshape(self.sz[0], self.sz[3], self.sz[2]).unsqueeze(1)), dim=1)
        return flow


class Initialize_Flow(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img, b):
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//b, W//b).cuda()
        coords1 = coords_grid(N, H//b, W//b).cuda()

        return coords0, coords1
    

class Conv1(nn.Module):
    def __init__(self, input_dim = 145):
        super(Conv1, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 1, padding=0, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x


class Conv3(nn.Module):
    def __init__(self, input_dim = 130):
        super(Conv3, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x
        

class RHWF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpuid[0]))
        self.args = args

        self.encoder = RHWF_Encoder(output_dim=96, norm_fn='instance')
        self.conv3 = Conv3(input_dim=130)
        
        if self.args.lev0:
            self.initialize_flow_4 = Initialize_Flow()
            self.transformer_0 = FocusFormer_Attention(96, 1, 96, 96)
            self.kernel_list_0 = [0, 9, 5, 3, 3, 3] #此处0表示GM全局
            self.pad_list_0    = [0, 4, 2, 1, 1, 1]
            sz = 32
            self.kernel_0 = 17
            self.pad_0 = 8
            self.conv1_0 = Conv1(input_dim=145)
            self.update_block_4 = GMA_update(self.args, sz)

        if self.args.lev1:
            self.initialize_flow_2 = Initialize_Flow()
            self.transformer_1 = FocusFormer_Attention(96, 1, 96, 96)
            self.kernel_list_1 = [5, 5, 3, 3, 3, 3]
            self.pad_list_1    = [2, 2, 1, 1, 1, 1]
            sz = 64
            self.kernel_1 = 9
            self.pad_1 = 4
            self.conv1_1 = Conv1(input_dim=81)
            self.update_block_2 = GMA_update(self.args, sz)
        
    def forward(self, image1, image2, iters_lev0 = 0, iters_lev1= 0, test_mode=False):

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0
        
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        image2_org = image2

        # feature network
        fmap1_32, fmap1_64 = self.encoder(image1) # [B, 96, 32, 32]
        fmap2_32, _        = self.encoder(image2) # [B, 96, 64, 64]
        
        four_point_disp = torch.zeros((image1.shape[0], 2, 2, 2)).cuda()
        four_point_predictions = []
        
        if self.args.lev0:
            coords0, coords1 = self.initialize_flow_4(image1, 4)
            coords0 = coords0.detach()
            sz = fmap1_32.shape
            self.sz = sz
            self.get_flow_now_4 = Get_Flow(sz)
            
            for itr in range(iters_lev0):
                if itr < 6:
                   fmap1, fmap2 = self.transformer_0(fmap1_32, fmap2_32, self.kernel_list_0[itr], self.pad_list_0[itr])
                else:
                   fmap1, fmap2 = self.transformer_0(fmap1_32, fmap2_32, 3, 1)
                    
                coords1 = coords1.detach()
                corr = F.relu(Correlation.apply(fmap1.contiguous(), fmap2.contiguous(), self.kernel_0, self.pad_0)) 
                b, h, w, _ = corr.shape
                corr_1 = F.avg_pool2d(corr.view(b, h*w, self.kernel_0, self.kernel_0), 2).view(b, h, w, 64).permute(0, 3, 1, 2)
                corr_2 = corr.view(b, h*w, self.kernel_0, self.kernel_0)
                corr_2 = corr_2[:,:,4:13,4:13].contiguous().view(b, h, w, 81).permute(0, 3, 1, 2)                                                  
                corr = torch.cat([corr_1, corr_2], dim=1)

                corr = self.conv1_0(corr)                                         
                flow = coords1 - coords0
                corr_flow = torch.cat((corr, flow), dim=1)
                corr_flow = self.conv3(corr_flow)             
                
                delta_four_point = self.update_block_4(corr_flow)
                four_point_disp =  four_point_disp + delta_four_point
                four_point_predictions.append(four_point_disp)
                coords1 = self.get_flow_now_4(four_point_disp, 4)
                
                if itr < (iters_lev0-1):
                    flow_med = coords1 - coords0
                    flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4              
                    flow_med = flow_med.detach()         
                    image2_warp = warp(image2_org, flow_med)
                    # save_img(torchvision.utils.make_grid((image2_warp+1)/2*255), './watch/' + 'test_img2_w_' + str(itr) + '.bmp')
                    fmap2_32_warp, _  = self.encoder(image2_warp)
                    fmap2_32 = fmap2_32_warp.float()               

        if self.args.lev1:
            flow_med = coords1 - coords0
            flow_med = F.upsample_bilinear(flow_med, None, [4, 4]) * 4            
            flow_med = flow_med.detach()
            image2_warp = warp(image2_org, flow_med)
            _, fmap2_64_warp = self.encoder(image2_warp)
            fmap2_64 = fmap2_64_warp.float()
            
            sz = fmap1_64.shape
            self.sz = sz
            self.get_flow_now_2 = Get_Flow(sz)
            
            coords0, coords1 = self.initialize_flow_2(image1, 2)
            coords0 = coords0.detach()
            coords1 = self.get_flow_now_2(four_point_disp, 2)
            
            for itr in range(iters_lev1):
                if itr < 6:
                    fmap1, fmap2 = self.transformer_1(fmap1_64, fmap2_64, self.kernel_list_1[itr], self.pad_list_1[itr])
                else:
                    fmap1, fmap2 = self.transformer_1(fmap1_64, fmap2_64, 3, 1)
                
                coords1 = coords1.detach()
                corr = F.relu(Correlation.apply(fmap1.contiguous(), fmap2.contiguous(), self.kernel_1, self.pad_1)).permute(0, 3, 1, 2)    
                
                corr = self.conv1_1(corr)   
                flow = coords1 - coords0
                corr_flow = torch.cat((corr, flow), dim=1)
                corr_flow = self.conv3(corr_flow)  
                
                delta_four_point = self.update_block_2(corr_flow)
                four_point_disp = four_point_disp + delta_four_point
                four_point_predictions.append(four_point_disp)
                coords1 = self.get_flow_now_2(four_point_disp, 2)
                
                if itr < (iters_lev1-1):
                    flow_med = coords1 - coords0
                    flow_med = F.upsample_bilinear(flow_med, None, [2, 2]) * 2
                    flow_med = flow_med.detach()
                    image2_warp = warp(image2_org, flow_med)
                    _, fmap2_64_warp = self.encoder(image2_warp)
                    fmap2_64 = fmap2_64_warp.float()
            
        if test_mode:
            return four_point_disp
        return four_point_predictions



