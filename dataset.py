import os,sys
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import numpy as np
import scipy.ndimage.morphology as morph

from img_utils import read3DImage


class PairedMaskDataset(Dataset):
    def __init__(self, root,mode='train',listnameA='data_list_deformed.txt',listnameB='data_list_undeformed.txt', edges=False):
        self.files_def=open(os.path.join(root,  mode + '/'+listnameA)).readlines()
        self.files_und = open(os.path.join(root,  mode + '/'+listnameB)).readlines()
        self.mode=mode
        self.edges=edges
    def __getitem__(self, index):
        eta=0.000001
        image_def_filename=self.files_def[index % len(self.files_def)].rstrip()
        image_def = read3DImage(image_def_filename)
        image_def_c0 = image_def == 0
        image_def_c1 = image_def == 127
        image_def_c2 = image_def == 254
        tumor_c2 = image_def == 64
        image_def_c2 = tumor_c2 + image_def_c2

        image_def_c0 = morph.distance_transform_edt(image_def_c0)
        image_def_c0/=np.max(image_def_c0)+eta
        image_def_c1 = morph.distance_transform_edt(image_def_c1)
        image_def_c1 /= np.max(image_def_c1)+eta
        image_def_c2 = morph.distance_transform_edt(image_def_c2)
        image_def_c2 /= np.max(image_def_c2)+eta

        image_und_filename = self.files_und[index % len(self.files_und)].rstrip()
        image_und = read3DImage(image_und_filename)

        image_und_c0 = image_und == 0
        image_und_c0 = np.logical_and(image_und_c0, np.logical_not(tumor_c2))
        image_und_c1 = image_und == 127
        image_und_c1=np.logical_and(image_und_c1, np.logical_not(tumor_c2))
        image_und_c2 = image_und == 254
        image_und_c2=np.logical_or(image_und_c2, tumor_c2)

        image_und_c0 = morph.distance_transform_edt(image_und_c0)
        image_und_c0 /= np.max(image_und_c0) + eta
        image_und_c1 = morph.distance_transform_edt(image_und_c1)
        image_und_c1 /= np.max(image_und_c1) + eta
        image_und_c2 = morph.distance_transform_edt(image_und_c2)
        image_und_c2 /= np.max(image_und_c2) + eta
        if self.edges:
            sketch_und_filename = image_und_filename.replace('_segm.png','_sketch.png')
            sketch_und = read3DImage(sketch_und_filename)
            sketch_und = morph.distance_transform_edt(1-sketch_und)
            sketch_und /= np.max(sketch_und) + eta
            sketch_def_filename = image_def_filename.replace('_segm.png', '_sketch.png')
            sketch_def = read3DImage(sketch_def_filename)
            sketch_def = morph.distance_transform_edt(1-sketch_def)
            sketch_def /= np.max(sketch_def) + eta
            image_und_channels = np.stack([image_und_c0, image_und_c1, image_und_c2,sketch_und])
            image_def_channels = np.stack([image_def_c0, image_def_c1, image_def_c2,sketch_def])
        else:
            image_und_channels = np.stack([image_und_c0, image_und_c1, image_und_c2])
            image_def_channels = np.stack([image_def_c0, image_def_c1, image_def_c2])
        image_und_channels= torch.tensor(image_und_channels).float()
        image_und_channels= F.interpolate(image_und_channels.unsqueeze(0), size=(40, 40,40), mode="trilinear").squeeze(0)
        image_def_channels = torch.tensor(image_def_channels).float()
        image_def_channels = F.interpolate(image_def_channels.unsqueeze(0), size=(40, 40,40), mode="trilinear").squeeze(0)
        tumor=tumor_c2
        tumor=torch.tensor(tumor).float()
        tumor=F.interpolate(tumor.unsqueeze(0).unsqueeze(0), size=(40,40,40), mode="nearest").squeeze(0)
        mask=torch.zeros_like(tumor.squeeze(0))
        mask[image_def_channels[0,:,:]==0]=1
        return {'image_def': image_def_channels.float(), 'image_und': image_und_channels.float(),'mask':mask.unsqueeze(0).float(),'tumor':tumor.float()}
    def __len__(self):
        return max(self.files_def.__len__(),self.files_und.__len__())


if __name__ == "__main__":
    print("")
