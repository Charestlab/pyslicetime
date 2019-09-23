import os
import numpy as np
from slicetime.nipype_interface import SliceTime

# get some test data
in_file = os.path.join('data', 'sub-1_run-1.nii.gz')

# define some parameters
tr_old = 2
tr_new = 1
n_slices = 32

# sliceorder needs to be 1-based (see fakeout below)
slicetimes = np.flip(np.arange(0, tr_old, tr_old/n_slices)).tolist()

st = SliceTime()
st.inputs.in_file = in_file
st.inputs.tr_old = tr_old
st.inputs.tr_new = tr_new
st.inputs.slicetimes = slicetimes
res = st.run()
