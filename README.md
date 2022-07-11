# AdapTable
This repository contains code and resources for the paper "_Exploring Few-Shot Adaptation of Language Models with Tables_" by Jun Shern Chan, Michael Pieler, Jonathan Jao, Jérémy Scheurer and Ethan Perez.

![Tables-to-tasks](/img/tables_to_tasks.png)

# AdapTable dataset

## Download
If you just want to download the dataset, you can download it from the HuggingFace Hub:
TODO

## Recreate

Install requirements:
```
conda create -n adaptable python=3.8
conda activate adaptable
python -m pip install -r requirements.txt
```

Recreating the dataset involves the following steps:
1. Download the `English-Language Relational Web Tables 2015` source tables from [WDC Web Table Corpus 2015](http://webdatacommons.org/webtables/2015/downloadInstructions.html).
2. Extract the files.
3. Convert the tables into tasks (.jsonl format).

Since the source tables are provided as 51 separate slices, we process each of the slices separately:
```bash
SLICE="00" # Repeat for each of 00, 01, 02 ... 50
# Download
wget http://data.dws.informatik.uni-mannheim.de/webtables/2015-07/englishCorpus/compressed/$SLICE.tar.gz
# Extract
tar -xvf $SLICE.tar.gz
# Convert
python tables_to_tasks.py --tarfile $SLICE.tar --outdir ./adaptable/ --max_source_files 10000
```

For convenience, we provide sbatch scripts for performing the the above steps in a parallelized manner on a SLURM system. To download and extract all 51 slices via 51 parallel batch jobs, simply run `bash download_and_process_all.sh`. (Caution: Will generate ~150GB and ~500k files)

# Reproducibility
Our main experiment setting uses [MetaICL](https://github.com/facebookresearch/MetaICL) for training and testing.

## Model weights
The model weights for our GPT2-large model fine-tuned on `AdapTable-5k` can be downloaded [here](https://drive.google.com/file/d/1Q1mh9rKxD6MX0lTD_okWEjINWRNfqhXY/view?usp=sharing).

## Training
Please follow the instructions on the [MetaICL](https://github.com/facebookresearch/MetaICL) repository for training.

## Evaluation
Please follow the instructions on the [MetaICL](https://github.com/facebookresearch/MetaICL) repository for test evaluation.