import os
import sys
import argparse
import glog as logger

import torch
import torch.nn as nn
import torch.utils.data
import torch.autograd
from deepgeom.datasets import PointCloudDenoiseDataset
from deepgeom.NPNet import NPNetVanilla, NPNetSingle, myMSELoss

from deepgeom.traintester import TrainTester
from deepgeom.utils import count_parameter_num


def main(args):

    # Load data
    kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        PointCloudDenoiseDataset(args.train_pkl, shuffle_point_order=args.shuffle_point_order),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        PointCloudDenoiseDataset(args.test_pkl, shuffle_point_order='no'), 
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    net = NPNetVanilla(
        MLP_dims1=(3,64,64,64,128),
        MLP_dims2=(128,512),
        #FC_dims=(1024,512,512),
        MLP_dims21 = (640,256,128),
        MLP_dims22 = (7,64,64,3)
    )

    logger.info('Number of parameters={}'.format(count_parameter_num(net.parameters())))
    solver = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss();
    runner = TrainTester(
        net=net, solver=solver, total_epochs=args.epoch,
        cuda=args.cuda, log_dir=args.log_dir, verbose_per_n_batch=args.verbose_per_n_batch
    )

    runner.run(train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn)
    torch.save(net.state_dict(),os.path.join(args.log_dir,'denoise_original_idea_test_worotation.pkl'))

    logger.info('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(sys.argv[0], description='NPNet Autoencoder')

    parser.add_argument('-0','--train-pkl',type=str,
                        help='path of the training pkl file')
    parser.add_argument('-1','--test-pkl',type=str,
                        help='path of the testing pkl file')
    parser.add_argument('-e','--epoch',type=int,default=300, # change the default setting
                        help='training epochs')#278
    parser.add_argument('--batch-size',type=int,default=1,
                        help='training batch size')
    parser.add_argument('--test-batch-size',type=int,default=1,
                        help='testing batch size')
    parser.add_argument('--lr',type=float,default=1e-4, # 1e-4 default
                        help='learning rate')
    parser.add_argument('--momentum',type=float,default=0.9, # 0.9 default
                        help='Solver momentum')
    parser.add_argument('--weight-decay',type=float,default=1e-6, # 1e-6 default
                        help='weight decay')
    parser.add_argument('--shuffle-point-order',type=str,default='no',
                        help='whether/how to shuffle point order (no/offline/online)')
    parser.add_argument('--log-dir',type=str,default='logs/tmp',
                        help='log folder to save training stats as numpy files')
    parser.add_argument('--verbose_per_n_batch',type=int,default=10,
                        help='log training stats to console every n batch (<=0 disables training log)')

    args = parser.parse_args(sys.argv[1:])
    args.script_folder = os.path.dirname(os.path.abspath(__file__))

    args.cuda = torch.cuda.is_available()

    print(str(args))
    sys.stdout.flush()

    main(args)
