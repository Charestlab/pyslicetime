# pyslicetime

## Slice-time interpolation for fMRI data.

cubic interpolation used to resample each voxel’s time-series data to a new rate such that the same time points are obtained
for all voxels. This can be useful if the trial duration is not evenly divisible by the TR, which then handles any jitter between
trial onsets and slice acquisition times.

The motivation for upsampling is to exploit the intrinsic jitter between the data acquisition and the experimental paradigm.

## matlab version:
An equivalent functionality exists in matlab, see:
https://github.com/kendrickkay/knkutils/blob/master/timeseries/tseriesinterp.m

## reference:
The logic of this procedure was introduced in the following pre-print:
https://www.biorxiv.org/content/10.1101/868455v1.full.pdf



