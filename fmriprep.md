build fmriprep with pyslicetime

docker build . -t fmriprep:pyslicetime
docker tag fmriprep:pyslicetime charestlab/fmriprep:pyslicetime
docker push charestlab/fmriprep:pyslicetime
singularity build Images/fmriprep-pyslicetime.simg docker://charestlab/fmriprep:pyslicetime