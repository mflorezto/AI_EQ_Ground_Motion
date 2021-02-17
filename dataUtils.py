# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mtspec import mtspec


def plot_waves_3C(ws, dt, t_max=20.0, Np=10, color='C0', chan='chan_first'):

    Nt = int(t_max / dt)
    if ws.shape[0] < Np:
 
    if chan == 'chan_first':
        ws_p = ws[:Np, :, :Nt]
    elif chan == 'chan_last':
        ws_p = np.transpose(ws, (0, 2, 1))
        ws_p = ws_p[:Np, :, :Nt]
    else:
        assert False, "Undefined channel type, it must be either chan_first or chan_last "

    tt = dt * np.arange(Nt)
    plt.figure()
    fig, axs = plt.subplots(Np, 3, sharex='col', sharey='row',
                            gridspec_kw={'hspace': 0, 'wspace': 0},
                            figsize=(14, 10))
    for i in range(Np):
        for j in range(3):
            wf = ws_p[i, j, :]
            if Np > 1:
                axs[i, j].plot(tt, wf, color=color)
            else:
                axs[j].plot(tt, wf, color=color)

    for ax in axs.flat:
        ax.label_outer()
        plt.show()



def make_bins(v, bins):
    if isinstance(bins, int):
        b = np.linspace(np.min(v), np.max(v), bins + 1)
        b[-1] = np.inf
        return b
    elif isinstance(bins, list):
        b = np.array([np.inf if b_ == 'inf' else b_ for b_ in bins])
        return b
    else:
        assert False, "Bins must either be specified as a list or an interger."


def make_ibins(v, nbins):
    assert isinstance(nbins, int), "nbins must be an interger."

    b_max = np.max(v)
    b_min = np.min(v)
    dbin = (b_max - b_min) / nbins
    b_max = b_max + 0.01 * dbin
    b = np.linspace(b_min, b_max, nbins + 1)
    return b


def get_bins_midpoint(bins):
    Nb = len(bins) - 1
    cb = np.zeros((Nb), dtype=np.float32)
    for i in range(Nb):
        cb[i] = (bins[i] + bins[i + 1]) / 2.0
    return cb


def rescale(v):
    v_min = v.min()
    v_max = v.max()
    dv = v_max-v_min
    vn = (v-v_min) / dv
    return vn



class SeisData(object):
    wfs = None
    vc_lst = None
    iv_lst = None
    vc_max = None
    vc_min = None
    v_names = None
    df_meta = None
    ebins = None
    bin_midpts = None

    # -------------------------------------------------

    def __init__(self, data_file, attr_file, batch_size, sample_rate, v_names, nbins_d, isel):
        self.data_file = data_file
        self.batch_size = batch_size

        if not isinstance(v_names, list):
            assert False, "Please supply names of conditional variables"

        self.v_names = v_names
        self.nbins_d = nbins_d
        # load data
        print('Loading data ...')
        wfs = np.load(data_file)
        print('Loaded samples: ', wfs.shape[0])
        # total number of training samples
        Ntrain = wfs.shape[0]
        self.Ntrain = Ntrain
        print('normalizing data ...')
        wfs_norm = np.max(np.abs(wfs), axis=2)
        wfs_norm = wfs_norm[:, :, np.newaxis]
        self.wfs = wfs / wfs_norm

        self.sample_rate = sample_rate
        self.dt = 1 / sample_rate

        df = pd.read_csv(attr_file)
        self.df_meta = df[self.v_names]

        self._set_vc_max()
        self._init_bins()
        self._init_vcond()


        Nsel = len(isel)

        self.wfs = self.wfs[isel, :, :]
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[isel, :])
        self.vc_lst = vc_b
        iv_b = []
        for iv in self.iv_lst:
            iv_b.append(iv[isel, :])
        self.iv_lst = iv_b

        self.ix = np.arange(Nsel)
        self.vc = np.hstack(self.vc_lst)

        self.iv_s = np.hstack(self.iv_lst)
        self.Ntrain = Nsel
        print('Number selected samples: ', Nsel)

        print('Class init done!')

    def to_real(self, vn, v_name):
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max-v_min
        v = vn*dv + v_min
        return v

    def to_syn(self, vr, v_name):
        v_min = self.vc_min[v_name]
        v_max = self.vc_max[v_name]
        dv = v_max-v_min
        vn = (vr-v_min) / dv
        return vn


    def _set_vc_max(self):
        self.vc_max = {}
        self.vc_min = {}
        for vname in self.v_names:
            v_max = self.df_meta[vname].max()
            self.vc_max[vname] = v_max
            v_min = self.df_meta[vname].min()
            self.vc_min[vname] = v_min

    def _init_bins(self):
        self.ebins = {}
        self.bin_midpts = {}
        for v_name in self.v_names:
            v = rescale(self.df_meta[v_name].to_numpy())
            self.ebins[v_name] = make_ibins(v, self.nbins_d[v_name])
            self.bin_midpts[v_name] = get_bins_midpoint(self.ebins[v_name])

    def _init_vcond(self):
        assert isinstance(self.ebins, dict), "Bins must be initialized"
        assert isinstance(self.bin_midpts, dict), "Bins must be initialized"


        self.iv_lst = []
        self.vc_lst = []
        for v_name in self.v_names:
            v = rescale(self.df_meta[v_name].to_numpy())
            ebins = self.ebins[v_name]
            print('---------', v_name, '-----------')
            print('min ' + v_name, self.df_meta[v_name].min())
            print('max ' + v_name, self.df_meta[v_name].max())

            iv = np.digitize(v, ebins) - 1
            iv = np.reshape(iv, (iv.shape[0], 1))
            self.iv_lst.append(iv)
            mid_bins = self.bin_midpts[v_name]
            fn = lambda ix: mid_bins[ix]
            vc = np.apply_along_axis(fn, 0, iv)
            print('vc shape', vc.shape)
            self.vc_lst.append(vc)

    def _get_condv(self):
        return self.iv_s

    def get_batch_vcond(self, iv_lst, Np=10):
        i_sel = self.iv_s[:, 0] == iv_lst[0]
        Nv = len(iv_lst)
        for ii in range(1, Nv):
            i_sel = i_sel & (self.iv_s[:, ii] == iv_lst[ii])

        assert i_sel.sum() > 0, "No Elements meet the conditions"
        ws = self.wfs[i_sel, :, :]
        print('batch_vcond shape', ws.shape)
        if ws.shape[0] > Np:
            # randomly select a batch of waves
            i_out = np.random.choice(Np, size=(Np,), replace=False)
            i_out.sort()
            return ws[i_out, :, :]
        else:
            return ws

    def _get_rand_idxs(self):
        ib = np.random.choice(self.ix, size=self.batch_size, replace=False)
        ib.sort()
        return ib

    def get_rand_batch(self):
        ib = self._get_rand_idxs()
        wfs_b = self.wfs[ib, :, :]
        wfs_b = wfs_b[:, :, :, np.newaxis]
        vc_b = []
        for v in self.vc_lst:
            vc_b.append(v[ib, :])

        return (wfs_b, vc_b)

    def get_rand_cond_v(self):
        vc_b = []
        for v in self.vc_lst:
            ii = self._get_rand_idxs()
            vc_b.append(v[ii, :])

        return vc_b

    def get_batch_size(self):
        return self.batch_size

    def __str__(self):
        return 'wfs data shape: ' + str(self.wfs.shape)

    def get_Ntrain(self):
        return self.Ntrain

    def get_Nbatches_tot(self):
        Nb_tot = np.floor(self.Ntrain / self.batch_size)
        return int(Nb_tot)

    def plot_waves_rand_3C(self, t_max=20.0, Np=10, color='C0', ):
        dt = self.dt
        (ws, _) = self.get_rand_batch()
        plot_waves_3C(ws, dt, t_max, Np=Np, color=color)

