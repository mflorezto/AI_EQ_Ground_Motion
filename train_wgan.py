# %%
import os
import json
import shutil

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# iimport librariess
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from imp import reload


# pytorch imports
import torch
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

# costum utilities
import dataUtils
import gan

# make sure class is reloaded everytime the cell is executed
reload(dataUtils_v1)
from dataUtils_v1 import SeisData, plot_waves_3C

reload(gan)
from gan import Discriminator, Generator

# ----- Input Parameters ------
config_d = {
    'data_file': '/scratch/mflorez/gmpw/train_set/train_vs30_pga/downsamp_5x_sel.npy',
    'attr_file': '/scratch/mflorez/gmpw/train_set/train_vs30_pga/wforms_table_sel.csv',
    'batch_size': 256,
    'noise_dim': 100,
    'epochs': 180,
    'sample_rate': 20.0,
    'frac_train': 0.8,
    'condv_names': ['dist', 'mag', 'vs30'],
    # ---- wassertddain gan configuration -----
    'critic_iter': 10,
    'gp_lambda': 10.0,
    'lr':  1e-1,
    # Adam optimizer parameters :
    'beta1': 0.0,
    'beta2': 0.9,
    # ----- output onfiiguration  -----------
    'out_dir': 'b1_lr1',
    'print_every': 200,
    'save_every': 1,
}

def init_gan_conf(conf_d):
    out_dir = conf_d['out_dir']

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fname = os.path.join(out_dir, 'config_d.json')
    with open(fname, 'w') as f:
        json.dump(conf_d, f, indent=4)
    models_dir = os.path.join(out_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    gan_file = os.path.join('./', 'gan.py')
    gan_out = os.path.join(out_dir, 'gan_models.py')
    shutil.copyfile(gan_file, gan_out)
    return out_dir, models_dir

OUT_DIR, MODEL_DIR = init_gan_conf(config_d)
print(OUT_DIR, MODEL_DIR)
print(config_d)

Ntot = 222378
frac = config_d['frac_train']
Nbatch = config_d['batch_size']

ix_all = np.arange(Ntot)
Nsel = int(Ntot*frac)
ix_train = np.random.choice(ix_all, size=Nsel, replace=False)
ix_train.sort()
ix_val = np.setdiff1d(ix_all, ix_train, assume_unique=True)
ix_val.sort()

sdat_train = SeisData(config_d['data_file'], attr_file=config_d['attr_file'], batch_size=Nbatch, sample_rate=config_d['sample_rate'], v_names=config_d['condv_names'], nbins_d=config_d['nbins_dict'], isel=ix_train)

print('total Train:', sdat_train.get_Ntrain())

sdat_val = SeisData(config_d['data_file'], attr_file=config_d['attr_file'], batch_size=Nbatch, sample_rate=config_d['sample_rate'], v_names=config_d['condv_names'], nbins_d=config_d['nbins_dict'], isel=ix_val)

print('total Validation:', sdat_val.get_Ntrain())
(wfs, i_vc) = sdat_val.get_rand_batch()
print('shape:', wfs.shape)


cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def noise(Nbatch, dim):
    m = 3
    return np.random.normal(size=[Nbatch, m, dim]).astype(
        dtype=np.float32)

# ------- TRAIN MODELS --------

num_epochs = config_d['epochs']
print_every = config_d['print_every']
z_size = config_d['noise_dim']
n_critic = config_d['critic_iter']
lr = config_d['lr']
beta1 = config_d['beta1']
beta2 = config_d['beta2']



D = Discriminator()
G = Generator(z_size=z_size)
print(D)
print(G)


if cuda:
    G.cuda()
    D.cuda()
    print('GPU available for training. Models moved to GPU')
else:
    print('Training on CPU.')

d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=[beta1,beta2])
g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=[beta1,beta2])


losses_train = []
losses_val = []

reg_lambda = config_d['gp_lambda']

Nbatch = sdat_train.get_batch_size()
N_train_btot = sdat_train.get_Nbatches_tot()
N_val_btot = sdat_val.get_Nbatches_tot()

print('Training Batches: ', N_train_btot)
print('Validation Batches: ', N_val_btot)

# ------ START TRAINING LOOP -------
for i_epoch in range(num_epochs):
    d_train_wloss = 0.0
    d_train_gploss = 0.0
    g_train_loss = 0.0
    d_val_wloss = 0.0
    d_val_gploss = 0.0
    g_val_loss = 0.0

    # ----- Training loop ------
    G.train()
    D.train()
    for i_batch in range(N_train_btot):
        for i_c in range(n_critic):
            ### ---------- DISCRIMINATOR STEP ---------------
            (data_b, i_vc) = sdat_train.get_rand_batch()
            real_wfs = torch.from_numpy(data_b).float()
            i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
            Nsamp = real_wfs.size(0)
            if cuda:
                real_wfs = real_wfs.cuda()
                i_vc = [i_v.cuda() for i_v in i_vc]

            d_optimizer.zero_grad()

            z = noise(Nbatch, z_size)
            z = torch.from_numpy(z).float()
            if cuda:
                z = z.cuda()
            fake_wfs = G(z,*i_vc)

            alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
            Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
            D_xp = D(Xp, *i_vc)
            Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
            grads = torch.autograd.grad(
                outputs=D_xp,
                inputs=Xp,
                grad_outputs=Xout,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            grads = grads.view(grads.size(0), -1)

            d_gp_loss = reg_lambda*((grads.norm(2, dim=1) - 1) ** 2).mean()
            d_w_loss = -torch.mean(D(real_wfs, *i_vc)) + torch.mean(D(fake_wfs, *i_vc))
            d_loss = d_w_loss + d_gp_loss
            d_loss.backward()
            d_optimizer.step()

        ### ---------- END DISCRIMINATOR STEP ---------------
        d_train_wloss = d_w_loss.item()
        d_train_gploss = d_gp_loss.item()

        ### -------------- TAKE GENERATOR STEP ------------------------
        g_optimizer.zero_grad()

        
        z = noise(Nbatch, z_size)
        z = torch.from_numpy(z).float()
        

        i_vg = sdat_train.get_rand_cond_v()
        i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
        

        if cuda:
            z = z.cuda()
            i_vg = [i_v.cuda() for i_v in i_vg]
        

        fake_wfs = G(z, *i_vg)
        
        g_loss = -torch.mean( D(fake_wfs, *i_vg) )

        
        g_loss.backward()
        
        g_optimizer.step()
        
        g_train_loss = g_loss.item()
        ### --------------  END GENERATOR STEP ------------------------
        

        losses_train.append((d_train_wloss, d_train_gploss, g_train_loss) )

        if i_batch % print_every == 0:
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                i_epoch + 1, num_epochs, d_loss.item(), g_loss.item()))
    # ----- End training epoch ------


    # ----------- Validation Loop --------------
    G.eval()
    D.eval()
    for i_batch in range(N_val_btot):

        (data_b, i_vc) = sdat_val.get_rand_batch()
        real_wfs = torch.from_numpy(data_b).float()
        i_vc = [torch.from_numpy(i_v).float() for i_v in i_vc]
    

        Nsamp = real_wfs.size(0)
    
        if cuda:
            real_wfs = real_wfs.cuda()
            i_vc = [i_v.cuda() for i_v in i_vc]

    
        z = noise(Nbatch, z_size)
        z = torch.from_numpy(z).float()
    

        if cuda:
            z = z.cuda()
    

        fake_wfs = G(z,*i_vc)

        alpha = Tensor(np.random.random((Nsamp, 1, 1, 1)))
    
        Xp = (alpha * real_wfs + ((1 - alpha) * fake_wfs)).requires_grad_(True)
    
        D_xp = D(Xp, *i_vc)
        Xout = Variable(Tensor(Nsamp,1).fill_(1.0), requires_grad=False)
    
        grads = torch.autograd.grad(
            outputs=D_xp,
            inputs=Xp,
            grad_outputs=Xout,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads.view(grads.size(0), -1)

        d_gp_loss = reg_lambda*((grads.norm(2, dim=1) - 1) ** 2).mean()
        d_w_loss = -torch.mean(D(real_wfs, *i_vc)) + torch.mean(D(fake_wfs, *i_vc))
        d_loss = d_w_loss + d_gp_loss

        d_val_wloss += d_w_loss.item()
        d_val_gploss += d_gp_loss.item()
        ### ---------- END DISCRIMINATOR STEP ---------------


        ### ---------- TAKE GENERATOR STEP ------------------------

    
        z = noise(Nbatch, z_size)
        z = torch.from_numpy(z).float()
        i_vg = sdat_val.get_rand_cond_v()
        i_vg = [torch.from_numpy(i_v).float() for i_v in i_vg]
        if cuda:
            z = z.cuda()
            i_vg = [i_v.cuda() for i_v in i_vg]
        fake_wfs = G(z, *i_vg)
        g_loss = -torch.mean( D(fake_wfs, *i_vg) )
        g_val_loss += g_loss.item()
            ### --------------  END GENERATOR STEP ------------------------
    
    d_val_wloss = d_val_wloss/N_val_btot
    d_val_gploss = d_val_gploss/N_val_btot
    g_val_loss = g_val_loss/N_val_btot
    losses_val.append((d_val_wloss, d_val_gploss, g_val_loss) )
    ### --------- End Validation -------
    
    # AFTER EACH EPOCH #
    if i_epoch % config_d['save_every'] == 0:
        fmodel = os.path.join(MODEL_DIR, 'model_G_epoch_'+str(i_epoch)+'.pth')
        torch.save({'state_dict': G.state_dict()}, fmodel)



ftrainl = os.path.join(OUT_DIR, 'train_losses.pkl')

with open(ftrainl, 'wb') as f:
    pkl.dump(losses_train, f)

fvall = os.path.join(OUT_DIR, 'val_losses.pkl')
with open(fvall, 'wb') as f:
    pkl.dump(losses_val, f)


print('Done!!')

