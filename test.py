import os
import numpy as np
import torch
import argparse
import options.options as options
from data import create_dataloader
from data.utils import save_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to yaml file')
    args = parser.parse_args()
    opt = options.parse(args.config, mode='valid')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('./results/model.pt').to(device)
    valid_loader = create_dataloader(opt)

    # logging
    cur_iter = 0
    total_iter = len(valid_loader) # batch_size=1

    model.eval()
    with torch.no_grad():
        for i, valid_data in enumerate(valid_loader):
            inputs = valid_data['LRs'].to(device) # 1NCHW
            outputs = model(inputs) # 1CHW
            assert outputs.shape[0]==1

            # bilinear version
            #outputs = torch.nn.Upsample(scale_factor=4, mode='bilinear',
            #    align_corners=False)(inputs[:,2,:,:,:])

            # save prediction
            video = valid_data['video'][0]
            key = valid_data['key'][0].item()
            img_SR = outputs.squeeze().permute(1,2,0) # HWC
            img_SR = img_SR[:,:,[2,1,0]] # BGR
            img_SR = img_SR.cpu().detach().numpy()
            
            dataset_name = opt['dataset']['name']
            vpath = os.path.join('./results', dataset_name, 'valid', video)
            save_img(vpath, key, img_SR, dataset_name=dataset_name) 

            cur_iter += 1
            if cur_iter % 10 == 0:
                print('Processing: [{}/{}]'.format(cur_iter, total_iter))
            

if __name__ == '__main__':
    main()


