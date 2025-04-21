from datapro import Simdata_pro, loading_data
import torch
from train import train_test
import os


class Config:
    def __init__(self):
        self.datapath = './datasets'
        self.kfold = 5
        self.batchSize = 128
        self.ratio = 0.2
        self.epoch = 8
        self.gcn_layers = 8
        self.gcn_layers1 = 2
        self.view = 3
        self.fm = 128
        self.fd = 128
        self.inSize = 128
        self.outSize = 128
        self.hiddenSize = 32
        self.PVN = 1/32
        self.Dropout = 0.1
        self.hdnDropout = 0.1
        self.fcDropout = 0.1
        self.num_heads1 = 2
        self.maskMDA = False
        self.device = torch.device('cuda')


def main():
    param = Config()
    for i in range(1):
        simData = Simdata_pro(param)
        train_data = loading_data(param)
        output_folder = f'./savemodel/{i+1}/'
        os.makedirs(output_folder, exist_ok=True)
        train_test(simData, train_data, param, state='test', output_folder=output_folder)


if __name__ == "__main__":
    main()
