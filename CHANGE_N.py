from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import numpy as np
import time

import torch
from torch.utils import data

from torch.optim import lr_scheduler

from torchvision import datasets, models

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


from torch.autograd import Variable

import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
torch.set_printoptions(precision=4)
np.set_printoptions(precision=4)
import itertools



def ch_gen(Size_area, Num_AP, Num_samples, PL_alpha=36., PL_const=0):
    ch_w_fading = []
    for i in range(Num_samples):
        loc = Size_area * (np.random.rand(Num_AP, 2) - 0.5)
        ch_w_temp_band = []
        dist_vec = loc.reshape(Num_AP, 1, 2) - loc
        dist_vec = np.linalg.norm(dist_vec, axis=2)
        dist_vec = np.maximum(dist_vec, 5)

        pu_ch_gain_db = - PL_const - PL_alpha * np.log10(dist_vec)
        pu_ch_gain = 10 ** (pu_ch_gain_db / 10)

        final_ch = np.maximum(pu_ch_gain, np.exp(-30))

        ch_w_fading.append(np.transpose(final_ch))
    return np.array(ch_w_fading)



def rand_alloc(num_sample, num_user, num_chan):
    mat_val = np.zeros((num_sample, num_user, num_chan))
    for i in range(num_sample):
        tot_iter = num_chan ** num_user
        rand_sel = np.random.randint(tot_iter)
        for j in range(num_user):
            sel_ind = (rand_sel // num_chan ** j) % num_chan
            mat_val[i, j, sel_ind] = 1
    return mat_val


def opt_alloc(chan, num_user, num_chan):
    chan_ori = np.exp(chan * chan_std + chan_mean)
    chan_ori = np.triu(chan_ori, 1)
    tot_int_val = 10000
    tot_alloc = 0
    tot_iter = num_chan ** num_user
    for i in range(tot_iter):
        mat_val = np.zeros((num_user, num_chan))

        for j in range(num_user):
            sel_ind = (i // num_chan ** j) % num_chan
            mat_val[j, sel_ind] = 1

        cur_int_val = np.sum(np.sum(np.matmul(chan_ori, mat_val) * mat_val, 1), 0)
        if cur_int_val < tot_int_val:
            tot_alloc = mat_val
            tot_int_val = cur_int_val


    return tot_alloc, tot_int_val


def heu_alloc(chan, num_user, num_chan):
    chan_ori = np.exp(chan * chan_std + chan_mean)
    chan_ori = np.triu(chan_ori, 1)
    chan_lower = 1000*np.tri(num_user, num_user)
    chan_temp  = chan_ori + chan_lower

    mat_val = np.zeros((num_user, num_chan))
    sel_val = 0
    chan_idx = 0
    round_count = True

    sel_idx_mat = np.zeros(num_user)

    ## Atleast two channels are assigned for each channel
    if num_user > num_chan * 2:
        for i in range(num_chan):
            result = np.where(chan_temp == np.amin(chan_temp))
            mat_val[result[0], i] = 1
            mat_val[result[1], i] = 1

            chan_temp[result[0], :] = 1000
            chan_temp[result[1], :] = 1000

            chan_temp[:, result[0]] = 1000
            chan_temp[:, result[1]] = 1000

            sel_idx_mat[result[0]] = 1
            sel_idx_mat[result[1]] = 1

        remain_usr = num_user - 2 * num_chan

        for i in range(remain_usr):
            cur_low_idx = 0
            cur_low_val = 100000
            idx_ch_temp = i % num_chan
            for j in range(num_user):
                mat_val_temp = mat_val.copy()
                if sel_idx_mat[j] == 0:
                    mat_val_temp[j, idx_ch_temp] = 1
                    temp_cur_val = np.sum(np.sum(np.matmul(chan_ori, mat_val_temp) * mat_val_temp, 1), 0)
                    if temp_cur_val < cur_low_val:
                        cur_low_idx = j
                        cur_low_val = temp_cur_val

            mat_val[cur_low_idx, idx_ch_temp] = 1
            sel_idx_mat[cur_low_idx] = 1

    else:
        sel_chan = 0
        remain_usr = num_user - 2 * sel_chan
        remain_ch = num_chan - sel_chan
        while remain_usr != remain_ch:
            result = np.where(chan_temp == np.amin(chan_temp))
            mat_val[result[0], sel_chan] = 1
            mat_val[result[1], sel_chan] = 1

            chan_temp[result[0], :] = 1000
            chan_temp[result[1], :] = 1000

            chan_temp[:, result[0]] = 1000
            chan_temp[:, result[1]] = 1000

            sel_idx_mat[result[0]] = 1
            sel_idx_mat[result[1]] = 1

            sel_chan += 1
            remain_usr = num_user - 2 * sel_chan
            remain_ch = num_chan - sel_chan

        for i in range(remain_usr):
            sel_flag = True
            for j in range(num_user):
                if (sel_idx_mat[j] == 0) & (sel_flag):
                    mat_val[j, sel_chan] = 1
                    sel_idx_mat[j] = 1
                    sel_flag = False
            sel_chan += 1

    tot_alloc = np.sum(np.sum(np.matmul(chan_ori, mat_val) * mat_val, 1), 0)

    return mat_val, tot_alloc





def cal_DL(chan, mat_val, chan_mean, chan_std):
    chan_ori = np.exp(chan * chan_std + chan_mean)
    chan_triu = np.triu(chan_ori, 1)
    cur_int_val = np.sum(np.sum(np.matmul(chan_triu, mat_val)*mat_val, 2), 1)
    return cur_int_val



class Net(nn.Module):
    def __init__(self, num_user, num_chan, hidden_dim, num_l):
        super(Net, self).__init__()
        self.num_user = num_user
        self.num_chan = num_chan
        self.hidden_dim = hidden_dim
        self.num_l = num_l

        ## List of linear layers
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(int((num_user) * (num_user-1) / 2), hidden_dim))
        for lll in range(num_l):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        ## List of dropout layers
        self.dropouts = nn.ModuleList()
        for lll in range(num_l):
            self.dropouts.append(nn.Dropout(0.1))

        ## List of Batch layers
        self.batches = nn.ModuleList()
        for lll in range(num_l):
            self.batches.append(nn.BatchNorm1d(hidden_dim))

        ## output for softmax and sigmoid
        self.out_RA = nn.Linear(hidden_dim, num_user * num_chan)

    def forward(self, x):
        z = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        for (layer, dropout, batch_norm) in zip(self.layers, self.dropouts, self.batches):
            # x = F.relu(nlayer(x))
            z = layer(z)
            z = batch_norm(z)
            z = F.relu(z)
            z = dropout(z)

        u = self.out_RA(z)
        u = u.view(-1, self.num_user, self.num_chan)
        out_RA = F.softmax(u, dim=2)
        return out_RA




def my_loss(output, chan, chan_mean, chan_std):
    chan_ori = torch.exp(chan*chan_std + chan_mean)
    chan_ori = torch.triu(chan_ori, diagonal=1)
    int_sum = torch.matmul(chan_ori, output)*output
    int_sum = torch.sum(torch.sum(int_sum, 2), 1)
    return torch.log(int_sum.mean())




class ChanDataset(Dataset):
    def __init__(self, loc_val):
        super(ChanDataset, self).__init__()
        self.x_data = torch.Tensor(loc_val)
        self.x_data = torch.triu(self.x_data, diagonal=1)
        temp_val = []
        idx = torch.triu_indices(*self.x_data[0].shape, 1)
        for i in range(loc_val.shape[0]):
            temp_val.append(self.x_data[i, idx[0], idx[1]])
        self.y_data = torch.stack(temp_val, 0)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return (x, y)





TOT_CCI_TOT = []

for jj in range(1):

    net_size = 50
    num_user = 5 * (jj+2)
    num_chan = 4
    tx_px = 10 ** (3.0)

    tot_sample_tr = int(10 ** 4)
    tot_sample_te = int(1 * 10 ** 1)

    loc_val_tr = ch_gen(net_size, num_user, tot_sample_tr, PL_alpha=38., PL_const=34.5)
    loc_val_te = ch_gen(net_size, num_user, tot_sample_te, PL_alpha=38., PL_const=34.5)

    loc_val_tr_db = np.log(loc_val_tr)
    loc_val_te_db = np.log(loc_val_te)

    chan_mean = np.mean(loc_val_tr_db)
    chan_std = np.std(loc_val_tr_db)

    loc_val_tr_norm = (loc_val_tr_db - chan_mean) / chan_std
    loc_val_te_norm = (loc_val_te_db - chan_mean) / chan_std

    tr_dataset = ChanDataset(loc_val_tr_norm)
    val_dataset = ChanDataset(loc_val_te_norm)

    train_dataloader = DataLoader(tr_dataset, batch_size=1024*8, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=tot_sample_te, shuffle=False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(num_user, num_chan, 3000, 6).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.1*1e-5)


    TOT_CCI_TEMP = []
    for i in range(2):
        net.train()
        cur_loss_tot = 0
        for index, (chan, chan_triu) in enumerate(train_dataloader):
            optimizer.zero_grad()
            cur_loss = my_loss(net(chan_triu.to(device)), chan.to(device), chan_mean, chan_std)
            cur_loss_tot += cur_loss
            cur_loss.backward()
            optimizer.step()
            #if index % 100 == 0:
            #    print("index = ", i, "curloss", cur_loss_tot.detach() / (index + 1))
            #    print("")

        net.eval()
        if i%25 == 0:
            for index, (chan_val, chan_triu_val) in enumerate(test_dataloader):
                dl_sel = net(chan_triu_val.to(device))
                chan_np = chan_val.cpu().detach().numpy()
                dl_per = cal_DL(chan_np, dl_sel.cpu().detach().numpy(), chan_mean, chan_std)
                print("DL per (during training) = ", 10*np.log10(np.mean(dl_per) / 2))


    ## Inference phase
    for index, (chan_val, chan_triu_val) in enumerate(test_dataloader):

        opt_val_tot = 0
        for jj in range(tot_sample_te):
            _, opt_val_temp = opt_alloc(chan_np[jj], num_user, num_chan)
            opt_val_tot += opt_val_temp
        print("Opt per = ", 10*np.log10(opt_val_tot / 2 / tot_sample_te))
        print("")

        TOT_CCI_TEMP.append(10*np.log10(opt_val_tot / 2 / tot_sample_te))


        heu_val_tot = 0
        for jj in range(tot_sample_te):
            _, heu_val_temp = heu_alloc(chan_np[jj], num_user, num_chan)
            heu_val_tot += heu_val_temp
        print("Heu per = ", 10*np.log10(heu_val_tot / 2 / tot_sample_te))
        print("")

        TOT_CCI_TEMP.append(10*np.log10(heu_val_tot / 2 / tot_sample_te))

        rand_alloc_val = rand_alloc(tot_sample_te, num_user, num_chan)
        rand_per = cal_DL(chan_np, rand_alloc_val, chan_mean, chan_std)
        print("Rand per = ", 10*np.log10(np.mean(rand_per) / 2))

        TOT_CCI_TEMP.append(10*np.log10(np.mean(rand_per) / 2))

        dl_sel = net(chan_triu_val.to(device))
        chan_np = chan_val.cpu().detach().numpy()
        dl_per = cal_DL(chan_np, dl_sel.cpu().detach().numpy(), chan_mean, chan_std)
        print("DL per = ", 10 * np.log10(np.mean(dl_per) / 2))

        TOT_CCI_TEMP.append(10 * np.log10(np.mean(dl_per) / 2))

        print("***"*30)
        print("***" * 30)
        print("***" * 30)


    TOT_CCI_TOT.append(np.array(TOT_CCI_TEMP))


print("CCI")
print(np.array(TOT_CCI_TOT))


