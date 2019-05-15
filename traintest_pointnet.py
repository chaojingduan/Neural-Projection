'''
The code is based on the paper of PointNet
'''
import os
import sys
import argparse
import glog as logger

import torch
import torch.utils.data
import torch.autograd
from deepgeom.datasets import PointCloudDataset
from deepgeom.pointnet import PointNetVanilla, PointNetTplMatch, PointNetAttentionPool, PointNetBilinearPool,\
     PointPairNet, BoostedPointPairNet, BoostedPointPairNet2, BoostedPointPairNetSuccessivePool, BoostedPointNetVanilla
from deepgeom.traintester import TrainTester
from deepgeom.utils import count_parameter_num


parser = argparse.ArgumentParser(sys.argv[0], description='PointNet Classification')

parser.add_argument('-0','--train-pkl',type=str,
                    help='path of the training pkl file')
parser.add_argument('-1','--test-pkl',type=str,
                    help='path of the testing pkl file')
parser.add_argument('-e','--epoch',type=int,default=100,
                    help='training epochs')
parser.add_argument('--batch-size',type=int,default=64,
                    help='training batch size')
parser.add_argument('--test-batch-size',type=int,default=32,
                    help='testing batch size')
parser.add_argument('--lr',type=float,default=0.1,
                    help='learning rate')
parser.add_argument('--momentum',type=float,default=0.5,
                    help='Solver momentum')
parser.add_argument('--weight-decay',type=float,default=1e-5,
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

#### Main
kwargs = {'num_workers':1, 'pin_memory':True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    PointCloudDataset(args.train_pkl, shuffle_point_order=args.shuffle_point_order),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    PointCloudDataset(args.test_pkl, shuffle_point_order='no'),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

test_loader.__iter__()

net = PointNetVanilla(
    MLP_dims=(3,64,64,64,128,1024),
    FC_dims=(1024,512,256,40)
)

logger.info('Number of parameters={}'.format(count_parameter_num(net.parameters())))
solver = torch.optim.SGD(net.parameters(),
                   lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
loss_fn = torch.nn.CrossEntropyLoss()
runner = TrainTester(
    net=net, solver=solver, total_epochs=args.epoch,
    cuda=args.cuda, log_dir=args.log_dir, verbose_per_n_batch=args.verbose_per_n_batch
)
runner.run(train_loader=train_loader, test_loader=test_loader, loss_fn=loss_fn)
logger.info('Done!')
