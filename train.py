import argparse
import options.options as options
from data import create_dataloader
from models import create_model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to yaml file')
    args = parser.parse_args()
    opt = options.parse(args.config, mode='train')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader = create_dataloader(opt)
    model = create_model(opt).to(device)

    # train the model
    beta1 = opt['train']['beta1']
    beta2 = opt['train']['beta2']
    lr = opt['train']['lr']
    num_epochs = opt['train']['num_epochs']

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

    # logging
    losses = []
    cur_iter = 0
    total_iter = len(train_loader) * num_epochs

    for epoch in range(num_epochs):
        for i, train_data in enumerate(train_loader):

            inputs = train_data['LRs'].to(device) # BNCHW
            targets = train_data['GT'].to(device) # BCHW
            outputs = model(inputs) # BCHW

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            cur_iter += 1
            if cur_iter % 1 == 0:
                losses.append(loss.item())
                print('Total loss: {:.4f} || MSE loss: {:.4f} || iter: [{}/{}]'\
                    .format(loss.item(), loss.item(), cur_iter, total_iter))

    torch.save(model, './results/model.pt')
    plt.plot(losses)
    plt.savefig('./results/loss.png')
    print('Training finished')



if __name__ == '__main__':
    main()
