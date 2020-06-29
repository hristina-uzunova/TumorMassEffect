import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import  Image
import numpy as np
import scipy
import matplotlib.pyplot as plt


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv3d') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm3d') != -1 or classname.find('InstanceNorm3d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm3d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm3d
    else:
        print('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def _compute_scaling_value(displacement):

    with torch.no_grad():
        scaling = 8
        norm = torch.norm(displacement / (2 ** scaling))

        while norm > 0.5:
            scaling += 1
            norm = torch.norm(displacement / (2 ** scaling))

    return scaling


def diffeomorphic_3d(displacement, grid, scaling=-1):
    if scaling < 0:
        scaling = _compute_scaling_value(displacement)
    displacement = displacement / (2 ** scaling)

    displacement = displacement.permute(0,4,1,2,3)

    for i in range(scaling):
        displacement_trans = displacement.permute(0,2,3,4,1)
        displacement = displacement + F.grid_sample(displacement, displacement_trans + grid, padding_mode='border', mode='bilinear')

    return displacement.permute(0,2,3,4,1)
"""
    Warp image with displacement
"""
def warp_image(image, displacement,mode='bilinear',):

    image_size = image.shape
    # image size [N,C,W,H (D)]
    image_size=image_size[2:]

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)
    grid_list=[]
    for i in range(image.shape[0]):
        grid_list.append(grid)
    grid=torch.cat(grid_list,dim=0)
    # warp image
    warped_image = F.grid_sample(image, displacement+grid,mode=mode)

    return warped_image

def warp_image_diffeomorphic(image, displacement,mode='bilinear',ret_displ=False):
    #displacement is defined by velocity

    image_size = image.shape
    # image size [N,C,W,H (D)]
    image_size=image_size[2:]

    grid = compute_grid(image_size, dtype=image.dtype, device=image.device)
    grid_list=[]
    for i in range(image.shape[0]):
        grid_list.append(grid)
    grid=torch.cat(grid_list,dim=0)
    # warp image
    displacement=diffeomorphic_3d(displacement, grid, scaling=20)
    warped_image = F.grid_sample(image, displacement+grid,mode=mode)
    if ret_displ:
        return warped_image, displacement
    else:
        return warped_image



def compute_grid(image_size, dtype=torch.float32, device='cpu'):
    dim = len(image_size)

    if(dim == 2):
        nx = image_size[0]
        ny = image_size[1]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)

        x = x.expand(ny, -1).transpose(0, 1)
        y = y.expand(nx, -1)

        x.unsqueeze_(0).unsqueeze_(3)
        y.unsqueeze_(0).unsqueeze_(3)

        return torch.cat((y, x), 3).to(dtype=dtype, device=device)

    elif(dim == 3):
        nz = image_size[2]
        ny = image_size[1]
        nx = image_size[0]

        x = torch.linspace(-1, 1, steps=nx).to(dtype=dtype)
        y = torch.linspace(-1, 1, steps=ny).to(dtype=dtype)
        z = torch.linspace(-1, 1, steps=nz).to(dtype=dtype)
        x, y, z = torch.meshgrid([x, y, z])
        # x = x.expand(ny, -1).expand(nz, -1, -1)
        # y = y.expand(nx, -1).expand(nz, -1, -1).transpose(1, 2)
        # z = z.expand(nx, -1).transpose(0, 1).expand(ny, -1, -1).transpose(0, 1)
        aff = torch.FloatTensor([[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]])
        grid = torch.nn.functional.affine_grid(aff, size=(1, 4, nz, ny, nx))


        return grid.to(dtype=dtype, device=device)
    else:
        print("Error " + str(dim) + "is not a valid grid type")
def diffusion_regularise_displacement(displacement):
    dim=len(displacement.shape)-2
    if dim==2:
        dx = (displacement[:,1:, 1:, :] - displacement[:,1:, :-1, :]).pow(2)
        dy = (displacement[:,1:, 1:, :] - displacement[:,:-1, 1:, :]).pow(2)
        return dx + dy
    if dim==3:
        dx= (displacement[:,1:, 1:, 1:, :] - displacement[:,1:, :-1, 1:, :]).pow(2)
        dy = (displacement[:,1:, 1:, 1:, :] - displacement[:,:-1, 1:, 1:, :]).pow(2)
        dz = (displacement[:,1:, 1:, 1:, :] - displacement[:,1:, 1:, :-1, :]).pow(2)
        return dx + dy + dz

def sparsity_regularise_displacement(displacement):
    return torch.abs(displacement)

class GaussianLayer(nn.Module):
    def __init__(self, c=1, sigma=3):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReplicationPad3d(10),
            nn.Conv3d(c, c, 21, stride=1, padding=0, bias=None, groups=c)
        )
        self.sigma = sigma
        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        n= np.zeros((21,21,21))
        n[10,10,10] = 1
        k = scipy.ndimage.gaussian_filter(n,sigma=self.sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

def calc_class_weights(train_set, image_name):
    weights=torch.zeros(train_set[0][image_name].shape[0])
    for img_nr in range(train_set.__len__()):
        data=train_set[img_nr]
        curr_img=data[image_name]
        n_pixels=curr_img.sum()
        for c in range(curr_img.shape[0]):
            weights[c]+=(curr_img[c].sum())
    weights /= weights.sum()
    weights=1-weights
    return weights

class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss,self).__init__()
        hx = np.array([1, 2, 1])
        hy = np.array([1, 2, 1])
        hz = np.array([1, 2, 1])
        hpx = np.array([1, 0, -1])
        hpy = np.array([1, 0, -1])
        hpz = np.array([1, 0, -1])
        gx = np.zeros((3, 3, 3))
        gy = np.zeros((3, 3, 3))
        gz = np.zeros((3, 3, 3))
        for m in range(0, 3):
            for n in range(0, 3):
                for k in range(0, 3):
                    gx[m, n, k] = hpx[m] * hy[n] * hz[k];
                    gy[m, n, k] = hx[m] * hpy[n] * hz[k];
                    gz[m, n, k] = hx[m] * hy[n] * hpz[k];
        self.kernel_x=gx
        self.kernel_y=gy
        self.kernel_z=gz

        self.conv_dx = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_dx.weight = nn.Parameter(torch.from_numpy(self.kernel_x).float().unsqueeze(0).unsqueeze(0))

        self.conv_dy = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_dy.weight = nn.Parameter(torch.from_numpy(self.kernel_y).float().unsqueeze(0).unsqueeze(0))

        self.conv_dz = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_dz.weight = nn.Parameter(torch.from_numpy(self.kernel_z).float().unsqueeze(0).unsqueeze(0))
    def forward(self,prediction,target):

        pred_dx = (self.conv_dx(prediction))
        pred_dy = (self.conv_dy(prediction))
        pred_dz = (self.conv_dz(prediction))
        target_dx = (self.conv_dx(target))
        target_dy = (self.conv_dy(target))
        target_dz = (self.conv_dz(target))

        grad_diff_x = (torch.abs(pred_dx) - torch.abs(target_dx))
        grad_diff_y = (torch.abs(pred_dy) - torch.abs(target_dy))
        grad_diff_z = (torch.abs(pred_dz) - torch.abs(target_dz))

        gdl = (grad_diff_x ** 2 + grad_diff_y ** 2+grad_diff_z**2).mean()
        return gdl


