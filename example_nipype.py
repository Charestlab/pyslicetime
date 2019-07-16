import os
from slicetime.nipype_interface import SliceTime

# get some test data
in_file = os.path.join('data', 'sub-1_run-1.nii.gz')

# define some parameters
tr_old = 2
tr_new = 1

# sliceorder needs to be 1-based (see fakeout below)
sliceorder = list(range(32, 0, -1))

st = SliceTime()
st.inputs.in_file = in_file
st.inputs.tr_old = tr_old
st.inputs.tr_new = tr_new
st.inputs.sliceorder = sliceorder
st.inputs.out_file = os.path.join('data', 'sub-1_run-1_slicetimed.nii.gz')
res = st.run()
