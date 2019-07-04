from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
from pointnet.dataset import PointCloudDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
# from wmp import snr, mse, chamfer_dist, pc_weight_epsilon
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import copy
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def snr(x,y):
    v = 20*np.log10(np.linalg.norm(x)/np.linalg.norm(x-y))
    return v

def mse(x,y):
    err = mean_squared_error(x,y)
    return err

def chamfer_dist(x,y):
    # y is denoised
    num_data = x.shape[0]
    pc_denoise_norm = np.sum(y**2, axis=-1)
    pc_pure_norm = np.sum(x**2, axis=-1)

    pc_matrix_dist = np.tile(pc_pure_norm, (num_data, 1)).T + np.tile(pc_denoise_norm, (num_data, 1)) - 2 * (x @ y.T)
    chamfer1 = np.sum(np.min(pc_matrix_dist, axis=0))
    chamfer2 = np.sum(np.min(pc_matrix_dist, axis=1))
    dist = (chamfer1 + chamfer2)/num_data
    return dist


def eval(model, test_dataset):
    num_pc = len(test_dataset)
    npoints = test_dataset.npoints
    pc_xyz_denoise = np.zeros((num_pc, npoints, 3))
    normal = np.zeros((num_pc, npoints, 3))
    snr_obj = np.zeros(num_pc)
    mse_obj = np.zeros(num_pc)
    chamfer_obj = np.zeros(num_pc)
    for i,data in tqdm(enumerate(testdataloader, 0)):
        _, _, pc_xyz_noise, _ = data

        pc_xyz_noise = pc_xyz_noise.transpose(2, 1)
        pc_xyz_noise = pc_xyz_noise.cuda()
        model = model.eval()
        output, _, _ = model(pc_xyz_noise)
        output = output.transpose(2, 1)

        for bs in range(opt.batchSize):
            normal_vec = output[bs, :, 0:-1].view(-1, 3)
            inter = output[bs, :, -1][:, None]
            pc_noise = pc_xyz_noise[bs].view(-1, 3)

            a_tp = torch.einsum("mk, mk -> m", normal_vec, pc_noise).view(-1, 1)
            pred = pc_noise - (a_tp * normal_vec - inter * normal_vec)
            pred = pred.detach().cpu().numpy()
            pc_noise = pc_noise.detach().cpu().numpy()
            pc_xyz_denoise[i*opt.batchSize+bs] = pred
            normal[i*opt.batchSize+bs] = normal_vec.detach().cpu().numpy()
            snr_obj[i*opt.batchSize+bs] = snr(pred, pc_noise)
            print(snr_obj[i*opt.batchSize+bs])
            mse_obj[i*opt.batchSize+bs] = mse(pred, pc_noise)
            chamfer_obj[i*opt.batchSize+bs] = chamfer_dist(pred, pc_noise)
    print("snr:", np.average(snr_obj))
    print("mse:", np.average(mse_obj))
    print("chamfer dist:", np.average(chamfer_obj))
    x = pc_xyz_denoise[0, :, 0]
    y = pc_xyz_denoise[0, :, 1]
    u = normal[0, :, 0]
    v = normal[0, :, 1]
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v)
    plt.savefig("denoised_1.png")
    x = pc_xyz_denoise[1, :, 0]
    y = pc_xyz_denoise[1, :, 1]
    u = normal[1, :, 0]
    v = normal[1, :, 1]
    fig, ax = plt.subplots()
    ax.quiver(x, y, u, v)
    plt.savefig("denoised_2.png")


parser = argparse.ArgumentParser()
parser.add_argument(
    '--batchSize', type=int, default=2, help='input batch size')
parser.add_argument(
    '--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument(
    '--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='results', help='output folder')
parser.add_argument('--model', type=str, default='', help='model path')
parser.add_argument('--path', type=str, required=True, help="dataset path")
parser.add_argument('--filename', type=str, required=True, help="dataset filename")
parser.add_argument('--class_choice', type=str, default='Chair', help="class_choice")
parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

dataset = PointCloudDataset(
    root=opt.path,
    filename=opt.filename)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

test_dataset = PointCloudDataset(
    root=opt.path,
    filename=opt.filename)
testdataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batchSize,
    shuffle=False,
    num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
print(opt.filename)
num_classes = 4
try:
    os.makedirs(opt.outf)
except OSError:
    pass

blue = lambda x: '\033[94m' + x + '\033[0m'

classifier = PointNetDenseCls(k=num_classes, feature_transform=opt.feature_transform)

if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))

optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)
classifier.cuda()

num_batch = len(dataset) / opt.batchSize
cosine_loss = torch.nn.CosineEmbeddingLoss()
mse_loss = torch.nn.MSELoss()

for epoch in range(opt.nepoch):
    scheduler.step()
    for i, data in enumerate(dataloader, 0):
        pc_xyz_denoise, normal_vec, pc_xyz_noise, label = data
        pc_xyz_denoise = pc_xyz_denoise.transpose(2, 1)
        normal_vec = normal_vec.transpose(2, 1)
        pc_xyz_noise = pc_xyz_noise.transpose(2, 1)
        pc_xyz_denoise, normal_vec, pc_xyz_noise, label = pc_xyz_denoise.cuda(), normal_vec.cuda(), pc_xyz_noise.cuda(), label.cuda()
        optimizer.zero_grad()
        classifier = classifier.train()
        output, trans, trans_feat = classifier(pc_xyz_noise) # output: (B, 4, N)
        output = output.view(-1, num_classes) # (B*N, 4)
        pc_xyz_denoise = pc_xyz_denoise.view(-1, 3)
        pc_xyz_noise = pc_xyz_noise.view(-1, 3)
        normal_vec = normal_vec.view(-1, 3)
        a_tp = torch.einsum("mk, mk -> m", output[:, 0:-1], pc_xyz_noise).view(-1, 1)
        pred = pc_xyz_noise - (a_tp * output[:, 0:-1] - output[:, -1][:, None] * output[:, 0:-1])
        loss1 = mse_loss(pred, pc_xyz_denoise)
        inter = torch.einsum("mk, mk -> m", pc_xyz_denoise, normal_vec)
        # loss2 = cosine_loss(output[:, 0:-1], normal_vec, target=torch.ones(len(normal_vec)).cuda())
        # loss2 = mse_loss(normal_vec, output[:, 0:-1])
        loss2 = mse_loss(inter, output[:, -1])
        loss = 0.3*loss1 + 0.7*loss2
        if opt.feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        print(loss1)
        print(loss2)
        # print('[%d: %d/%d] train loss: %f' % (epoch, i, num_batch, loss.item()))

    torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.outf, opt.class_choice, epoch))
    if (epoch+1) == 20:
        eval(classifier, test_dataset)
        print(loss2)

# i = 0
# npoints = dataset.npoints
# pc_xyz_noise = test_dataset.noise_xyz[i, :, :][:, None]
# pc_xyz_denoise = test_dataset.denoise_xyz[i, :, :][:, None]
# normal_vec = test_dataset.uvw[i, :, :][:, None]
# pc_xyz_noise = pc_xyz_noise.transpose(0, 2, 1)
# pc_xyz_noise = torch.tensor(pc_xyz_noise).cuda()
# print(pc_xyz_noise.shape)
# output, _, _ = classifier(pc_xyz_noise)
# print(output.shape)
# output = output.view(-1, num_classes)

# pc_xyz_noise = pc_xyz_noise.view(-1, 3)
# a_tp = torch.einsum("mk, mk -> m", output[:, 0:-1], pc_xyz_noise).view(-1, 1)
# pred = pc_xyz_noise - (a_tp * output[:, 0:-1] - output[:, -1][:, None] * output[:, 0:-1])
# pred = pred.detach().cpu().numpy()
# output = output.view(-1, npoints, num_classes).detach().cpu().numpy()
# x = pred[:, 0]
# y = pred[:, 1]
# u = output[:, :, 0]
# v = output[:, :, 1]
# fig, ax = plt.subplots()
# ax.quiver(x, y, u, v)
# plt.savefig("denoised_1024.png")

# x = pc_xyz_denoise[:, :, 0]
# y = pc_xyz_denoise[:, :, 1]
# u = normal_vec[:, :, 0]
# v = normal_vec[:, :, 1]
# fig, ax = plt.subplots()
# ax.quiver(x, y, u, v)
# plt.savefig("groundtruth_1024.png")
# print("snr:", snr(pred, pc_xyz_noise.detach().cpu().numpy()))