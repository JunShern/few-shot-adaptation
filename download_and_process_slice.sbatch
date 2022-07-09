#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --mem=32GB

#SBATCH --job-name=build_adaptable

## This places the standard output and standard error into the same file, in this case slurm_<job_id>.out 
#SBATCH --output=slurm_%A.out

## Load conda environment
module purge
module load anaconda3/2020.07
source ~/.bashrc
conda activate $HOME/.conda/envs/adaptable

SLICE=$1
BASE_DIR=$HOME/git/AdapTable/
DOWNLOAD_DIR=$BASE_DIR/download
OUT_DIR=$BASE_DIR/dataset

mkdir -p $DOWNLOAD_DIR && cd $DOWNLOAD_DIR
wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/$SLICE.tar.gz
tar -xvf $SLICE.tar.gz

mkdir -p $OUT_DIR && cd $OUT_DIR
python tables_to_tasks.py --tarfile $DOWNLOAD_DIR/$SLICE.tar --outdir $OUT_DIR --max_source_files 10000
echo "EXPORT COMPLETE. Exported to $OUT_DIR/$SLICE/"