from __future__ import print_function
import argparse
import itertools
import os,sys
from math import log10
import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from probabilistic_UNet import ProbabilisticUnet, l2_regularisation
from dataset import PairedMaskDataset
from net_utils import calc_class_weights,warp_image, warp_image_diffeomorphic, diffusion_regularise_displacement, GaussianLayer, sparsity_regularise_displacement
from img_utils import save3DImage
# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataroot', type=str, default='/share/data_tiffy2/uzunova/3DTumorDeformations/Lists', help='root directory of the dataset')
parser.add_argument('--resultsroot', type=str, default='/share/data_tiffy2/uzunova/3DTumorDeformations/Results', help='root directory of the dataset')
parser.add_argument('--experiment_type', type=str, default='TumorDeformations', help='root directory of the dataset')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--nEpochs', type=int, default=500)
parser.add_argument('--valPr', type=int, default=0.1)
parser.add_argument('--startRegWeight', type=int, default=0.002)
parser.add_argument('--batchSize', type=int, default=1)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--traintype',type=str, default='def',help="Types:[normal, adv, cycle, def]")
opt = parser.parse_args()

# torch.manual_seed(20)
device = torch.device('cuda', opt.device)
print('===> Loading datasets')
inputDir = os.path.join(opt.dataroot,opt.experiment_type)
outputDir=os.path.join(opt.resultsroot,opt.experiment_type)
if not os.path.exists(outputDir):
    os.makedirs(outputDir)
valOutDir=os.path.join(outputDir,'Val')
if not os.path.exists(valOutDir):
    os.makedirs(valOutDir)

train_set = PairedMaskDataset(inputDir)
val_set=PairedMaskDataset(inputDir)

num_train=len(train_set)
indices=list(range(num_train))
split=int(np.floor(opt.valPr*num_train))
np.random.seed(32)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

training_data_loader = DataLoader(dataset=train_set, num_workers=2, batch_size=opt.batchSize, sampler=train_sampler)
validation_data_loader = DataLoader(dataset=val_set, num_workers=2, batch_size=1, sampler=valid_sampler)
print('===> Building model')
with torch.cuda.device(opt.device):
    netG=ProbabilisticUnet(input_channels=4,dim=3, num_filters=[32,64,128], latent_dim=15, no_convs_fcomb=3, beta=0.000001)
    criterionGAN = nn.CrossEntropyLoss()
    criterionL1=nn.L1Loss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0001)
    print('---------- Networks initialized -------------')
    netG=netG.to(device)
    Gauss = GaussianLayer(sigma=3).to(device)
    criterionGAN = criterionGAN.to(device)
    criterionL1=criterionL1.to(device)

def validate(epoch):
    netG.eval()
    with torch.no_grad():
        for i,batch in enumerate(validation_data_loader):
            if i<2:
                images_und = batch['image_und'].to(device)
                images_def = batch['image_def'].to(device)
                mask=batch['mask'].to(device)
                tumor=batch['tumor'].to(device)
                netG(torch.cat((images_def,tumor),dim=1),None, training=False)

                displ=netG.sample()
                displ = displ.permute(0, 2, 3,4, 1)
                #computed_displ=displ
                prediction, computed_displ=warp_image_diffeomorphic(images_def, displ, mode='bilinear', ret_displ=True)
                #prediction=warp_image(images_def,displ,mode='bilinear')
                lossGAN = criterionL1(prediction, images_und)
                prediction=torch.argmax(prediction[:,:,:,:,:],dim=1)
                images_und=torch.argmax(images_und[:,:,:,:,:],dim=1)
                images_def = torch.argmax(images_def[:,:,:,:,:], dim=1)
                save3DImage(prediction.data[0,:,:,:].cpu().numpy().astype('uint8'),outputDir+"/Val/prediction_"+str(i)+".nii")
                save3DImage(images_und.data[0,:,:,:].cpu().numpy().astype('uint8'),outputDir + "/Val/realund_" + str(i) +".nii")
                save3DImage(images_def.data[0,:,:,:].cpu().numpy().astype('uint8'),outputDir + "/Val/realdef_" + str(i) +".nii")

                fig = plt.figure()
                x, y = np.meshgrid(np.arange(0, images_def.data.cpu().numpy()[0, :, :].shape[0]),
                                   np.arange(0, images_def.data.cpu().numpy()[0, :, :].shape[1]))
                x_plt = plt.imshow(
                    computed_displ.data.cpu().numpy()[0,15, :, :, 0] ** 2 + computed_displ.data.cpu().numpy()[0, 15,:, :, 1] ** 2)
                plt.quiver(x[::5,::5], y[::5,::5], computed_displ.data.cpu().numpy()[0, 15,:, :, 0][::5,::5], computed_displ.data.cpu().numpy()[0, 15,:, :, 1][::5,::5])
                cbar = fig.colorbar(x_plt)
                cbar.minorticks_on()
                fig.savefig(outputDir + "/Val/displ_" + str(i) + ".png")
        print("===> Valid Loss: "+str(lossGAN.item()))

def train_def(epoch,reg_weight):
    netG.train()
    tumor_reg_weight=100
    reg_loss_mean=0
    elbo_mean=0
    if epoch % 50 == 0 and reg_weight<0.001:
        reg_weight *= 2
        print("Reg weight changed: "+str(reg_weight))
    for iteration, batch in enumerate(training_data_loader, 1):
        images_und = batch['image_und'].to(device)
        images_def = batch['image_def'].to(device)
        tumor=batch['tumor'].to(device)
        netG(torch.cat((images_def,tumor),dim=1),torch.cat((images_und,tumor),dim=1))
        elbo = netG.elbo(images_def, images_und)
        displ_reg_loss=sparsity_regularise_displacement(netG.displ) # velocity reg
        tumor_cut=(torch.max(tumor)-tumor)
        soft_mask=tumor_reg_weight*Gauss(tumor_cut).detach()
        soft_mask=soft_mask+1 # mehr gewichten
        soft_mask = torch.stack((soft_mask[:,0,...],soft_mask[:,0,...],soft_mask[:,0,...]), dim=4)
        displ_reg_loss=displ_reg_loss*soft_mask
        tumor_reg=diffusion_regularise_displacement(netG.displ)
        loss = -elbo +reg_weight*(tumor_reg.mean()+displ_reg_loss.mean())
        reg_loss_mean+=reg_weight*(tumor_reg.sum()+displ_reg_loss.mean()).item()
        elbo_mean+=-elbo.item()
        print("Reg loss: " + str(reg_loss_mean))
        print("Elbo:" + str(elbo_mean))
        optimizerG.zero_grad()
        loss.backward()
        optimizerG.step()
        save3DImage(netG.computed_displ.data[0, :, :, :, 1].cpu().numpy(),
                    outputDir + "/Val/displ_" + str(iteration) + ".nii")
        print("===> Epoch[{}]({}/{}): Loss G: {:.6f}".format(
            epoch, iteration, len(training_data_loader), loss.item()))
    return reg_weight, reg_loss_mean/(iteration+1), elbo_mean/(iteration+1)


def test(epoch):
    print('Testing...')


with torch.cuda.device(opt.device):
    train_losses = []
    val_losses = []
    if opt.mode=="train":
        reg_weight=opt.startRegWeight
        for epoch in range(1, opt.nEpochs + 1):
            if opt.traintype == 'def':
                # only the G has changed
                reg_weight, reg_loss, elbo=train_def(epoch,reg_weight)
                val_losses.append(reg_loss)
                train_losses.append(elbo)
            if epoch % 1 == 0:
                torch.save(netG.state_dict(), outputDir + '/netG_epoch_' + str(epoch) + '.pth')
                validate(epoch)
            plt.figure()
            train_ax, = plt.plot(np.arange(0, epoch, 1), val_losses, label='Reg')
            val_ax, = plt.plot(np.arange(0, epoch, 1), train_losses, label='Elbo')
            plt.ylim((0,1))
            plt.legend(handles=[train_ax, val_ax])
            plt.savefig(outputDir + "/Val/curves_epoch.png")

    elif opt.mode=="test":
        test(5)

