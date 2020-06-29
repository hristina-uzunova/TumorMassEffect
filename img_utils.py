import os,sys
from glob import glob
import numpy as np
import SimpleITK as sitk
import scipy.misc as misc
from PIL import Image
from HelperTools.utils import readImage, saveImage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as morph
from HelperTools.utils import crop_BB, normalizeImage
from skimage import feature, filters, morphology, segmentation

def generateSketch(img, sigma=1, low_threshold=40, high_threshold=80):
    edges_in=sitk.CannyEdgeDetection(sitk.GetImageFromArray(img.astype(float)),lowerThreshold=low_threshold,upperThreshold=high_threshold)#,variance=[sigma, sigma, sigma])
    magnitude=sitk.SobelEdgeDetection(sitk.GetImageFromArray(img.astype(float)))
    edges_in=sitk.GetArrayFromImage(edges_in)
    magnitude=sitk.GetArrayFromImage(magnitude)
    magnitude=(magnitude-np.min(magnitude))/(np.max(magnitude)-np.min(magnitude))
    return (255*edges_in*magnitude)

def get_percent_of_voxels(img,percent):
    sorted=np.sort(img[img>0])
    length=sorted.shape[0]
    return sorted[int(round(length*percent))]

def segmentVentricles(img, ventr_thresh=0.2):
    thresh_ventr = get_percent_of_voxels(img,ventr_thresh)
    ventr = np.zeros_like(img)
    ventr[img < thresh_ventr] = 1
    ventr[img == 0] = 0
    ventr = morph.binary_fill_holes(ventr)
    brain_tissue = np.zeros_like(img)
    brain_tissue[img > 0] = 1
    brain_mask_eroded = morph.binary_erosion(brain_tissue, iterations=15)
    ventr = ventr * brain_mask_eroded
    brain_tissue = brain_tissue - ventr
    segm_img = np.maximum(ventr.astype('uint8'), brain_tissue.astype('uint8') * 2)
    segm_img *= 127
    return segm_img

def save3DImage(img, path, inf_img=None):
    img = sitk.GetImageFromArray(img)
    if inf_img is not None:
        img.SetDirection(inf_img.GetDirection())
        img.SetOrigin(inf_img.GetOrigin())
        img.SetSpacing(inf_img.GetSpacing())
    sitk.WriteImage(img,path)

def read3DImage(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def normalizeImage(img, minValue=0,maxValue=255, valueType='uint8'):
    img=(img-np.min(img))/(np.max(img)-np.min(img))
    img=img*(maxValue-minValue)+minValue
    return img.astype(valueType)

def crop_BB(img, offset=3, crop_coords=None):
    if crop_coords is None:
        bb_coords = np.where(img > 0)
        x_min = np.min(bb_coords[0])
        x_min = max(x_min-offset, 0)
        x_max = np.max(bb_coords[0])
        x_max = min(x_max+offset, img.shape[0])
        y_min = np.min(bb_coords[1])
        y_min = max(y_min-offset, 0)
        y_max = np.max(bb_coords[1])
        y_max = min(y_max+offset, img.shape[1])
        if (len(img.shape)==3):
            z_min = np.min(bb_coords[2])
            z_min = max(z_min - offset, 0)
            z_max = np.max(bb_coords[2])
            z_max = min(z_max + offset, img.shape[2])
            return img[x_min:x_max, y_min:y_max, z_min:z_max],(x_min,x_max,y_min,y_max,z_min,z_max)
        return img[x_min:x_max, y_min:y_max], (x_min,x_max,y_min,y_max)
    else:
        x_min = crop_coords[0]
        x_min = max(x_min, 0)
        x_max = crop_coords[1]
        x_max = min(x_max, img.shape[0])
        y_min = crop_coords[2]
        y_min = max(y_min, 0)
        y_max = crop_coords[3]
        y_max = min(y_max, img.shape[1])
        if (len(img.shape) == 3):
            z_min = crop_coords[4]
            z_min = max(z_min, 0)
            z_max = crop_coords[5]
            z_max = min(z_max , img.shape[2])
            return img[x_min:x_max, y_min:y_max, z_min:z_max], crop_coords
        return img[x_min:x_max, y_min:y_max], crop_coords