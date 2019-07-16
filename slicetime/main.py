import nibabel as nib
import numpy as np
from scipy.stats import rankdata
from slicetime.tseriesinterp import tseriesinterp


def run_slicetime(inpath, outpath, slicetimes=None, tr_old=2,
                  tr_new=1,
                  time_dim=2,
                  offset=0):
    """[summary]

    Args:
        inpath ([type]): [description]
        outpath ([type]): [description]
        slicetimes ([type]): [description]
        tr_old (int, optional): [description]. Defaults to 2.
        tr_new (int, optional): [description]. Defaults to 1.
        time_dim (int, optional): [description]. Defaults to 2.
        offset (int, optional): [description]. Defaults to 0.
    """

    data_pointer = nib.load(inpath)
    data = data_pointer.get_data()

    # what's in it?
    x, y, z, n_times = data.shape

    upscale = tr_old/tr_new
    new_n_times = int(np.ceil(n_times*upscale))
    epi_data_corr = np.zeros((x, y, z, new_n_times))

    # sliceorder needs to be 1-based (see fakeout below)
    if slicetimes is None:
        slicetimes = np.flip(np.arange(0, tr_old, tr_old/z))

    sliceorder = rankdata(slicetimes, method='dense')

    numsamples = None

    for slice_i in range(z):
        this_slice_ts = data[:, :, slice_i, :]
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

        # slice time corrected interpolated data
        epi_data_corr[:, :, slice_i, :] = new_slice_ts

    # write the file out
    # first we modify the TR in the header,
    # and the number of resulting volumes
    hdr = data_pointer.header
    hdr['pixdim'][4] = tr_new
    hdr['dim'][4] = new_n_times

    # then we write to disk
    corr_img = nib.Nifti1Image(epi_data_corr, data_pointer.affine, hdr)
    nib.save(corr_img, outpath)
