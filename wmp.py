import numpy as np
import os
import copy
from scipy.io import loadmat
from scipy.sparse import csr_matrix
from itertools import cycle 
from sklearn.metrics import mean_squared_error
import argparse
# from collections import namedtuple

parser = argparse.ArgumentParser(description='Denoising')
parser.add_argument('--path', type=str, default='dataset/')
parser.add_argument('--object', type=str, default='bottle',
                    help="target object to be denoised")
parser.add_argument('--num_pc', type=int, default=10,
                    help='number of pieces of objects')
parser.add_argument('--lmd', type=float, default=0.67)
parser.add_argument('--eta', type=float, default=0.05)

args = parser.parse_args()


# to do
def write_xyz(fout, coords, title="", atomtypes=("A",)):
    """ write a xyz file from file handle
    Writes coordinates in xyz format. It uses atomtypes as names. The list is
    cycled if it contains less entries than there are coordinates,
    One can also directly write xyz data which was generated with read_xyz.
    >>> xx = read_xyz("in.xyz")
    >>> write_xyz(open("out.xyz", "w"), *xx)
    Parameters
    ----------
    fout : an open file
    coords : np.array
        array of coordinates
    title : title section, optional
        title for xyz file
    atomtypes : iteratable
        list of atomtypes.
    See Also
    --------
    read_xyz
    """
    if type(fout == str):
        fout = open(fout, "w")
    fout.write("%d\n%s\n" % (coords.size / 3, title))
    for x, atomtype in zip(coords.reshape(-1, 3), cycle(atomtypes)):
        fout.write("%s %.18g %.18g %.18g\n" % (atomtype, x[:, 0], x[:, 1], x[:, 2]))


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


def pc_weight_epsilon(pc_xyz, epsilon_scale):
    num_data , num_dim = pc_xyz.shape # N by d
    print('The number of the input signal is %d\n' % num_data)
    print('The dimension of the input signal is %d\n' % num_dim)
    pc_xyz_col = np.sum(pc_xyz ** 2, axis=1)
    pc_matrix_dist = np.tile(pc_xyz_col, (num_data, 1)) + np.tile(pc_xyz_col, (num_data, 1)).T - 2 * (pc_xyz @ pc_xyz.T) # square of dist
    epsilon_default = np.sqrt(np.mean(pc_matrix_dist)) # the default epsilon is the root of mean value of distance matrix
    epsilon = epsilon_default * epsilon_scale
    print('The e NN graph you would like to build is e = %.10f\n' % epsilon)

    pc_matrix_preserve = pc_matrix_dist
    pc_matrix_dist[pc_matrix_dist > epsilon] = 0


    pc_dist_kernel = np.exp(-pc_matrix_dist/epsilon)
    pc_dist_kernel[pc_dist_kernel == 1] = 0 # convert ii entry to 0
    pc_dist_kernel -= np.diag(np.diag(pc_dist_kernel))

    eps = 2.2204e-16
    pc_weight_epsilon_unit = csr_matrix(pc_dist_kernel/(np.sum(pc_dist_kernel, axis=1)[:, None] + eps)) # normalized
    pc_weight_epsilon_sym = csr_matrix(pc_dist_kernel)

    num_neigh = csr_matrix.count_nonzero(pc_weight_epsilon_sym[0, :])

    print('The number of neighbors of first point is %d\n' % num_neigh)
    return pc_weight_epsilon_sym , pc_weight_epsilon_unit , epsilon


def tpe(pc_xyz_noise, pc_weight_unit, pc_weight_sym):
    pc_smooth_plane = np.zeros_like(pc_xyz_noise)
    pc_smooth_patch = np.zeros_like(pc_xyz_noise)
    pc_avg_plane = pc_weight_unit @ pc_xyz_noise
    num_data = pc_xyz_noise.shape[0]
    normal_vec = np.zeros((num_data, 3))
    inter = np.zeros((num_data, 1))

    for ii in range(num_data):
        pc_neigh_plane_ii = pc_xyz_noise[csr_matrix.nonzero(pc_weight_unit[ii])[-1]].T
        pc_neigh_patch_ii = pc_xyz_noise[csr_matrix.nonzero(pc_weight_sym[ii])[-1]].T
        pc_neigh_mat_ii_tr = np.zeros_like(pc_neigh_plane_ii)[None, :, :]
        pc_neigh_mat_ii_tr[0, :, :] = pc_neigh_plane_ii
        pc_neigh_mat_ii = np.transpose(pc_neigh_mat_ii_tr, [1, 0, 2])
        pc_avg_ii = pc_avg_plane[ii][:, None]
        weight_ii = pc_weight_unit[ii] # sum to 1
        weight_ii = np.asarray(weight_ii[:, csr_matrix.nonzero(pc_weight_unit[ii])[-1]].todense())
        weight_mat_ii = np.tile(weight_ii, (3, 3, 1))
        M_ii = np.sum(pc_neigh_mat_ii * pc_neigh_mat_ii_tr * weight_mat_ii, axis=-1) - pc_avg_ii @ pc_avg_ii.T # k: # of xyz(3), m:1, n: # of nonzero
        D,V = np.linalg.eig(M_ii) # D: eigenvalue V: eigenvector
        D_sort = np.sort(D)[::-1]
        index = np.argsort(D)[::-1]
        
        V_sort = V[:,index]
        n_ii = V_sort[:, -1] # min eigenvector (normal vector)
        c_ii = pc_avg_ii.T @ n_ii # intercept
        normal_vec[ii , :] = n_ii
        inter[ii , :] = c_ii
        pc_neigh_plane_proj_ii = pc_xyz_noise[ii , :].T - (n_ii @ pc_xyz_noise[ii , :] - c_ii) * n_ii
        pc_neigh_patch_proj_ii = pc_neigh_patch_ii - n_ii[:, None] * np.tile((n_ii @ pc_neigh_patch_ii - np.tile(c_ii, (1 , pc_neigh_patch_ii.shape[-1]))), (3 , 1))
        weight_patch = np.asarray(pc_weight_sym[ii, csr_matrix.nonzero(pc_weight_sym[ii])[-1]].todense())
        pc_smooth_plane[ii] = pc_neigh_plane_proj_ii.T
        pc_smooth_patch[csr_matrix.nonzero(pc_weight_sym[ii])[-1], : ] = pc_smooth_patch[csr_matrix.nonzero(pc_weight_sym[ii])[-1], : ] + (weight_patch * pc_neigh_patch_proj_ii).T
    pc_smooth_patch = pc_smooth_patch/(np.sum(pc_weight_sym, axis=1))

    pc_xyz_denoise_wmp = np.array(pc_smooth_plane * 0.6 + pc_smooth_patch * 0.4)
    return pc_xyz_denoise_wmp, normal_vec, inter

def wmp(pc_xyz_noise, pc_weight_unit, normal_vec, inter, lmd=2/3, eta=0.05):
    num_data = pc_xyz_noise.shape[0]
    pc_xyz_denoise = copy.deepcopy(pc_xyz_noise)

    for ii in range(num_data):
        pc_neigh_ii = csr_matrix.nonzero(pc_weight_unit[ii])[-1]
        mu = 0
        t = np.zeros(3)
        num_neigh = len(pc_neigh_ii)
        eta = np.log10(num_neigh)/num_neigh # might be changed
        sum_prod = 0
        for jj in pc_neigh_ii:
            proj = pc_xyz_denoise[ii] - (normal_vec[ii] @ pc_xyz_denoise[ii] - inter[ii]) # argmin_t||p-t||_2^2
            # t_prev = t
            t = 1 / (lmd*pc_weight_unit[ii, jj]+eta) * (lmd*pc_weight_unit[ii, jj] * pc_xyz_denoise[ii]
                + eta * (proj - mu/eta))
            
            # print((lmd*pc_weight_unit[ii, jj]+eta))
            # print(lmd*pc_weight_unit[ii, jj] * pc_xyz_noise[ii])
            # print(eta * (proj - mu/eta))
            mu += 2 * eta * (t - proj)
            sum_prod += pc_weight_unit[ii, jj] * t
        pc_xyz_denoise[ii] = 1 / (1+lmd) * (pc_xyz_noise[ii] + lmd * sum_prod)
    return pc_xyz_denoise

path = args.path
num_pc = args.num_pc
obj = args.object
lmd = args.lmd
eta = args.eta

snr_obj = np.zeros(num_pc)
mse_obj = np.zeros(num_pc)
chamfer_obj = np.zeros(num_pc)
noise_path = os.path.join(path, obj.capitalize(), obj + "_mat_noise")
pure_path = os.path.join(path, obj.capitalize(), obj + "_mat")

for jj in range(1, num_pc):
    print(jj)

    file_pc_noise = str('%s_noise%d.mat' % (obj, jj)) # change this for different dataset
    pc_struct_noise = loadmat(os.path.join(noise_path, file_pc_noise))
    pc_xyz_noise = pc_struct_noise['pc_xyz_noise']
    
    file_pc_pure = str('%s%d.mat' % (obj, jj)) # change this for different dataset
    pc_struct_pure = loadmat(os.path.join(pure_path, file_pc_pure))
    pc_xyz_unit = pc_struct_pure['pc_xyz_unit']
    num_data_ii = pc_xyz_noise.shape[0]
    
    # weight matrix
    epsilon_scale_ii = 0.006 * 15000/num_data_ii
    pc_weight_sym , pc_weight_unit , radius = pc_weight_epsilon(pc_xyz_noise, epsilon_scale_ii)

    pc_xyz_denoise_wmp, normal_vec, inter = tpe(pc_xyz_noise, pc_weight_unit, pc_weight_sym)
    snr_obj[jj - 1] = snr(pc_xyz_unit, pc_xyz_denoise_wmp)
    print(snr_obj)
    mse_obj[jj - 1] = mse(pc_xyz_unit, pc_xyz_denoise_wmp)
    print(mse_obj)
    chamfer_obj[jj - 1] = chamfer_dist(pc_xyz_unit, pc_xyz_denoise_wmp)
    print(chamfer_obj)

    # np.save("pc_xyz_noise.npy", pc_xyz_noise)
    # np.save("pc_weight_unit.npy", pc_weight_unit)
    # np.save("normal_vec.npy", normal_vec)
    # np.save("inter.npy", inter)
    # assert False


    # pc_xyz_noise = np.load("pc_xyz_noise.npy")
    # pc_weight_unit = np.load("pc_weight_unit.npy")
    # normal_vec = np.load("normal_vec.npy")
    # inter = np.load("inter.npy")


    pc_xyz_denoise_wmp = wmp(pc_xyz_noise, pc_weight_unit, normal_vec, inter, lmd, eta)
    snr_obj[jj - 1] = snr(pc_xyz_unit, pc_xyz_denoise_wmp)
    print(snr_obj)
    # write_xyz("bottle_denoise%d.xyz" % jj, pc_xyz_denoise_wmp)
    mse_obj[jj - 1] = mse(pc_xyz_unit, pc_xyz_denoise_wmp)
    print(mse_obj)
    chamfer_obj[jj - 1] = chamfer_dist(pc_xyz_unit, pc_xyz_denoise_wmp)
    print(chamfer_obj)
    assert False




