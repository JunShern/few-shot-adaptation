#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

#SBATCH --job-name=build_adaptable

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out 
#SBATCH --output=slurm_%A.out

## First we ensure a clean environment by purging the current one
module purge

## Load Anaconda
module load anaconda3/2020.07
source ~/.bashrc
conda activate $HOME/.conda/envs/metaicl

## Just log environment stats for diagnostics
myquota
nvidia-smi
which python

WDC_VERSION="v5"

SLICE=$1

cd $SCRATCH/MetaICL/data/wdc-$WDC_VERSION
wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/$SLICE.tar.gz
tar -xvf $SLICE.tar.gz

cd $HOME/git/MetaICL/wdc_setup
python tables_to_tasks.py --tarfile /scratch/jc11431/MetaICL/data/wdc-tars/$SLICE.tar --outdir /scratch/jc11431/MetaICL/data/wdc-$WDC_VERSION/ --max_source_files 10000
echo "EXPORT COMPLETE. Exported to /scratch/jc11431/MetaICL/data/wdc-$WDC_VERSION/$SLICE/"