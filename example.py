import numpy as np
import nibabel as nib
import os
from tseriesinterp import tseriesinterp
from make_image_stack import make_image_stack
from matplotlib import pyplot as plt
import seaborne as sns

tr_old = 2
tr_new = 1
time_dim = 2
offset = 0

# some example raw data
epirunfile = os.path.join('data', 'sub-1_run-1.nii.gz')

# load the run's epi in
epi_img = nib.load(epirunfile)
epi_data = epi_img.get_data()

# what's in it?
x, y, z, n_times = epi_data.shape

# sliceorder needs to be 1-based (see fakeout below)
sliceorder = list(range(z, 0, -1))
numsamples = None

for slice_i in range(z):
    this_slice_ts = epi_data[:, :, slice_i, :]
    # this_slice_ts.shape = (64, 64, 216) # time dim = 2

    this_slice_order = sliceorder[slice_i]
    max_slice = np.max(sliceorder)
    fakeout = -(((1-this_slice_order)/max_slice) * tr_old) - offset
    # print(fakeout)

    new_slice_ts = tseriesinterp(this_slice_ts,
                                 tr_old,
                                 tr_new,
                                 time_dim,
                                 numsamples,
                                 fakeout,
                                 wantreplicate=True,
                                 interpmethod='pchip')
    if slice_i == 0:
        new_n_times = new_slice_ts.shape[2]
        epi_data_corr = np.zeros((x, y, z, new_n_times))

    epi_data_corr[:, :, slice_i, :] = new_slice_ts

m = make_image_stack(np.mean(epi_data_corr, axis=3))
plt.image(m)
plt.show()
