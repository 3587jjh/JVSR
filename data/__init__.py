import torch
from torch.utils.data import DataLoader


def create_dataloader(opt):
    if opt['dataset']['name'] == 'REDS':
        from data.REDS_dataset import REDSDataset as D
    else:
        raise NotImplementedError
    dataset = D(opt)
    batch_size = opt['dataset']['batch_size']

    if opt['mode'] == 'train':
        return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    else:
        return DataLoader(dataset, batch_size=1, shuffle=False)
