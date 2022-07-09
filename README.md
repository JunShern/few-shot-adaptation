# AdapTable
This repository contains code and resources for the paper "_Exploring Few-Shot Adaptation of Language Models with Tables_" by Jun Shern Chan, Michael Pieler, Jonathan Jao, Jérémy Scheurer and Ethan Perez.

# AdapTable dataset

## Download
If you just want to download the dataset, you can download it from the HuggingFace Hub:
TODO

## Recreate
Recreating the dataset yourself involves just a few simple steps:
1. Download the `English-Language Relational Web Tables 2015` source tables from [WDC Web Table Corpus 2015](http://webdatacommons.org/webtables/2015/downloadInstructions.html).
    ```
    wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/$SLICE.tar.gz
    ```
2. Extract the files:
    ```
    tar -xvf $SLICE.tar.gz
    ```
3. Convert the tables into tasks (.jsonl format):
    ```
    python tables_to_tasks.py --tarfile /scratch/jc11431/MetaICL/data/wdc-tars/$SLICE.tar --outdir /scratch/jc11431/MetaICL/data/wdc-$WDC_VERSION/ --max_source_files 10000
    ```

For convenience, we provide sbatch scripts for performing the all the above steps in a parallelized manner on a SLURM system. To download and extract all 51 slices via 51 parallel batch jobs, simply run `bash download_and_process_all.sh`. (Caution: Will generate ~150GB and ~500k files)

## Data subsets

#### Clustering

#### Single-website tables