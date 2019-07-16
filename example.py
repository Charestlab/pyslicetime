import numpy as np
import nibabel as nib
import os
from slicetime.main import run_slicetime
from slicetime.make_image_stack import make_image_stack

tr_old = 2
tr_new = 1
time_dim = 2
offset = 0

# some example raw data
in_file = os.path.join('data', 'sub-1_run-1.nii.gz')
out_file = os.path.join('data', 'sub-1_run-1_slicetimed.nii.gz')

# sliceorder needs to be 1-based (see fakeout below)
sliceorder = list(range(32, 0, -1))

run_slicetime(
    inpath=in_file,
    outpath=out_file,
    sliceorder=sliceorder,
    tr_old=tr_old,
    tr_new=tr_new,
)
