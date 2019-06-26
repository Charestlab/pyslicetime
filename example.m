% needs spm12 for reading in the nifti file
addpath(genpath(fullfile('/home','adf','charesti','Documents','Software','spm12')));

% needs knkutils
addpath(genpath(fullfile('/home','adf','charesti','Documents','Software','knkutils')));

tr_old = 2;
tr_new = 1;
time_dim = 4;
offset = 0;
numsamples =[];
% some example raw data
epirunfile = fullfile('data', 'sub-1_run-1.nii.gz');

% load the run's epi in
epi_img = spm_vol(epirunfile);
epi_data = spm_read_vols(epi_img);

% what's in it?
[x, y, z, n_times] = size(epi_data);

% sliceorder needs to be 1-based (see fakeout below)
sliceorder = z:-1:1;
 
epistemp = cast([],class(epi_data));
for slice_i =1:z
    this_slice_ts = epi_data(:, :, slice_i, :);
    
    this_slice_order = sliceorder(slice_i);
    max_slice = max(sliceorder);
    fakeout = -(((1-this_slice_order)/max_slice) * tr_old) - offset;
    
    temp0 = tseriesinterp(this_slice_ts,tr_old,tr_new,time_dim,numsamples, ...
                                    fakeout, ...
                                    1,'pchip');
                                
    epistemp(:,:,slice_i,:) = temp0;                         
end

figure; 
imagesc(makeimagestack(mean(epistemp,4)));
   