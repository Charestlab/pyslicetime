# mkdir /tmp/data/reports && \
sudo docker run -ti --rm=false \
    -v /tmp/data:/tmp/data \
    -e FMRIPREP_REGRESSION_SOURCE=/tmp/data/fmriprep_bold_truncated \
    -e FMRIPREP_REGRESSION_TARGETS=/tmp/data/fmriprep_bold_mask \
    -e FMRIPREP_REGRESSION_REPORTS=/tmp/data/reports \
    --entrypoint="py.test" fmriprep:pyslicetime \
    /src/fmriprep/ \
    -svx --doctest-modules --ignore=/src/fmriprep/docs --ignore=setup.py