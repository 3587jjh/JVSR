import torch
import torch.nn as nn
from models.utils import backward_warp


class FlowEstimator(nn.Module):

    def __init__(self):
        super().__init__()

        self.coarse_flow_layer = nn.Sequential(
            nn.Conv2d(6, 24, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(4)
        )
        self.fine_flow_layer = nn.Sequential(
            nn.Conv2d(11, 24, 5, 2, 2),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 8, 3, 1, 1),
            nn.Tanh(),
            nn.PixelShuffle(2)
        )

    # predict flow imgA -> imgB
    def forward(self, A, B): #BCHW
        x = torch.cat([A, B], dim=1)
        coarse_flow = self.coarse_flow_layer(x)
        warpedB = backward_warp(B, coarse_flow)

        x = torch.cat([A, B, coarse_flow, warpedB], dim=1)
        fine_flow = self.fine_flow_layer(x)
        flow = coarse_flow + fine_flow
        return flow # B2HW



class Alignment(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.N = opt['N']
        self.L = opt['arch']['alignment']['level']
        self.flow_estimator = FlowEstimator()
        self.downsample = nn.AvgPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.pyramid_layers_bottom_up = []
        self.pyramid_layers_top_down = []

        for l in range(0, self.L-1):
            cur_channel = 3 if l==0 else 24
            self.pyramid_layers_bottom_up.append(
                nn.Sequential(
                    nn.Conv2d(cur_channel, 24, 5, 2, 2),
                    nn.ReLU(),
                    nn.Conv2d(24, 24, 3, 1, 1)
                )
            )
            self.pyramid_layers_top_down.append(
                nn.Conv2d(cur_channel+24, 24, 1, 1)
            )
        self.pyramid_layers_bottom_up = nn.ModuleList(self.pyramid_layers_bottom_up)
        self.pyramid_layers_top_down = nn.ModuleList(self.pyramid_layers_top_down)


    def forward(self, inputs): # BNCHW
        key = self.N
        aligned_list = []

        for i in range(0, 2*self.N+1):
            A = inputs[:,key,:,:,:]
            B = inputs[:,i,:,:,:]
            flow = self.flow_estimator(A, B) 
            warped_list = []

            for l in range(0, self.L):
                warped_list.append(backward_warp(B, flow))
                if l+1 == self.L:
                    break
                B = self.pyramid_layers_bottom_up[l](B)
                flow = self.downsample(flow)/2

            aligned = warped_list[-1]
            for l in range(self.L-2, -1, -1):
                aligned = self.upsample(aligned)
                aligned = torch.cat([aligned, warped_list[l]], dim=1)
                aligned = self.pyramid_layers_top_down[l](aligned)
            aligned_list.append(aligned)
        return aligned_list
            


class Fusion(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.N = opt['N']
        self.sigmoid = nn.Sigmoid()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d((3 if opt['arch']['alignment']['level']==1 else 24)*(2*self.N+1), 24, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(24, 24, 3, 1, 1)
        )
                
    def forward(self, aligned_list): # list of BCHW
        key = self.N
        # consider temporal attention
        attended_list = []
        for i in range(0, 2*self.N+1):
            attention = torch.sum(aligned_list[i]*aligned_list[key], dim=1)
            attention = self.sigmoid(attention).unsqueeze(1) # B1HW
            attended_list.append(aligned_list[i]*attention)
           
        fused_feature = torch.cat(attended_list, dim=1) # BCHW
        fused_feature = self.fusion_conv(fused_feature)
        return fused_feature



class BasicBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(24, 24, 3, 1, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(24, 24, 3, 1, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class Reconstruction(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.num_block = opt['arch']['reconstruction']['block']
        self.res_blocks = []
        for i in range(self.num_block):
            self.res_blocks.append(BasicBlock())
        self.res_blocks = nn.ModuleList(self.res_blocks)

    def forward(self, x):
        for i in range(self.num_block):
            x = self.res_blocks[i](x)
        return x
        


class JVSRBase(nn.Module):
    
    def __init__(self, opt):
        super().__init__()

        self.alignment = Alignment(opt)
        self.fusion = Fusion(opt)
        self.reconstruction = Reconstruction(opt)

        self.key = opt['N']
        self.r = opt['scale']
        self.upsample1 = nn.Sequential(
            nn.Conv2d(24, 3*self.r*self.r, 3, 1, 1),
            nn.PixelShuffle(self.r)
        )
        self.upsample2 = nn.Upsample(scale_factor=self.r, mode='bilinear', 
            align_corners=False)
        self.conv = nn.Conv2d(3, 3, 3, 1, 1)

    def forward(self, inputs): # BNCHW
        x = self.alignment(inputs) # list of BCHW
        x = self.fusion(x) # BCHW
        x = self.reconstruction(x) # BCHW

        x = self.upsample1(x)
        y = self.upsample2(inputs[:,self.key,:,:,:]) 
        x = self.conv(x+y)
        return x
        
                
