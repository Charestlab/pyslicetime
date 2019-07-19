#!/bin/bash
DATADIR="/castles/nr/projects/2018/charesti-start/data/pyslicetime"
SUBLABEL="1"
NTHREADS=${SLURM_NTASKS:-4}
MAXMEMMB=${SLURM_MEM_PER_NODE}
echo -e "\n"
echo "Starting fmriprep.."
echo "subject: $SUBLABEL"
echo "threads: $NTHREADS"
echo "memory: $MAXMEMMB"
echo "working directory: $TMPDIR"
echo "data directory: $DATADIR"
echo -e "\n"
singularity run -B $DATADIR:/data -B $TMPDIR:/work -B $TMPDIR:/tmp -c -e fmriprep-pyslicetime.simg \
    /data/BIDS/ \
    /data/BIDS/derivatives/ \
    participant \
    --participant-label $SUBLABEL \
    --work-dir /work \
    --fs-license-file /data/fs-license/license.txt \
    --nthreads $NTHREADS \
    --mem-mb $MAXMEMMB \
    --fs-no-reconall \
    --upsample-tr 0.5
