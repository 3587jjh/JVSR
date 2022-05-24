import argparse
import options.options as options
from data import create_dataloader
from models import create_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='path to yaml file')
    args = parser.parse_args()
    opt = options.parse(args.config, mode='train')
   
    train_loader = create_dataloader(opt)
    model = create_model(opt)










    #for train_data in train_loader:
    #    inputs = train_data['LRs'] # BNCHW
    #    targets = train_data['GT'] # BCHW
    #    outputs = model(inputs)







if __name__ == '__main__':
    main()
