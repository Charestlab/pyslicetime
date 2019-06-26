import numpy as np
import nibabel as nib
import os
from tseriesinterp import tseriesinterp
from make_image_stack import make_image_stack
from matplotlib import pyplot as plt
import seaborn as sns

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

upscale = tr_old/tr_new
new_n_times = int(np.ceil(n_times*upscale))
epi_data_corr = np.zeros((x, y, z, new_n_times))

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

    epi_data_corr[:, :, slice_i, :] = new_slice_ts

m = make_image_stack(np.mean(epi_data_corr, axis=3))
sns.heatmap(m)
plt.show()
