import torch
import torch.nn.functional as F


def backward_warp(x, flow):
    # x = BCHW, flow = BHW2
    B,C,H,W = x.shape
    assert flow.shape[0]==B and flow.shape[1]==H and flow.shape[2]==W
    grid_y, grid_x = torch.meshgrid(torch.arange(0,H), torch.arange(0,W))
    grid = torch.stack((grid_y, grid_x), 2).float() # HW2
    
    vgrid = grid + flow # BHW2
    vgrid_y = 2*vgrid[:,:,:,0]/(H-1)-1.0
    vgrid_x = 2*vgrid[:,:,:,1]/(W-1)-1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3) # BHW2
    
    output = F.grid_sample(x, vgrid_scaled, align_corners=False) # BCHW
    return output

