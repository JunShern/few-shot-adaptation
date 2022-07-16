# few-shot-adaptation
This repository contains code and resources for the paper "_Exploring Few-Shot Adaptation of Language Models with Tables_" by Jun Shern Chan, Michael Pieler, Jonathan Jao, Jérémy Scheurer and Ethan Perez.

![Tables-to-tasks](/img/tables_to_tasks.png)

## AdapTable dataset

### Download
Our datasets are available on the HuggingFace Hub. We provide the complete dataset `AdapTable-full` as well as the various sub-distributions discussed in our paper, for a total of 57 dataset options.

To download a dataset, simply `pip install datasets` and download the dataset using `load_dataset`:
```python
from datasets import load_dataset

distribution_names = [
    # Full dataset
    "MicPie/adaptable_full",
    # 5k random tasks from full dataset
    "MicPie/adaptable_5k",
    # Filtered to 1 task per website
    "MicPie/adaptable_unique",
    #  Single website tasks
    "MicPie/adaptable_baseball.fantasysports.yahoo.com",
    "MicPie/adaptable_bulbapedia.bulbagarden.net",
    "MicPie/adaptable_cappex.com",
    "MicPie/adaptable_cram.com",
    "MicPie/adaptable_dividend.com",
    "MicPie/adaptable_dummies.com",
    "MicPie/adaptable_en.wikipedia.org",
    "MicPie/adaptable_ensembl.org",
    "MicPie/adaptable_gamefaqs.com",
    "MicPie/adaptable_mgoblog.com",
    "MicPie/adaptable_mmo-champion.com",
    "MicPie/adaptable_msdn.microsoft.com",
    "MicPie/adaptable_phonearena.com",
    "MicPie/adaptable_sittercity.com",
    "MicPie/adaptable_sporcle.com",
    "MicPie/adaptable_studystack.com",
    "MicPie/adaptable_support.google.com",
    "MicPie/adaptable_w3.org",
    "MicPie/adaptable_wiki.openmoko.org",
    "MicPie/adaptable_wkdu.org",
    # Single cluster tasks
    "MicPie/adaptable_cluster00", "MicPie/adaptable_cluster01", "MicPie/adaptable_cluster02", "MicPie/adaptable_cluster03", "MicPie/adaptable_cluster04", "MicPie/adaptable_cluster05", "MicPie/adaptable_cluster06", "MicPie/adaptable_cluster07", "MicPie/adaptable_cluster08", "MicPie/adaptable_cluster09", "MicPie/adaptable_cluster10", "MicPie/adaptable_cluster11", "MicPie/adaptable_cluster12", "MicPie/adaptable_cluster13", "MicPie/adaptable_cluster14", "MicPie/adaptable_cluster15", "MicPie/adaptable_cluster16", "MicPie/adaptable_cluster17", "MicPie/adaptable_cluster18", "MicPie/adaptable_cluster19", "MicPie/adaptable_cluster20", "MicPie/adaptable_cluster21", "MicPie/adaptable_cluster22", "MicPie/adaptable_cluster23", "MicPie/adaptable_cluster24", "MicPie/adaptable_cluster25", "MicPie/adaptable_cluster26", "MicPie/adaptable_cluster27", "MicPie/adaptable_cluster28", "MicPie/adaptable_cluster29", "MicPie/adaptable_cluster-noise", 
    # Manual-rated tasks
    "MicPie/adaptable_rated-low", "MicPie/adaptable_rated-medium", "MicPie/adaptable_rated-high",
]

# Get the 5k sample dataset
dataset = load_dataset('MicPie/adaptable_5k')
```

We provide a demo of loading and inspecting tasks from the dataset at `adaptable_dataset_demo.ipynb`. Click the badge below to open the notebook and start exploring the dataset directly in Colab!

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JunShern/few-shot-adaptation/blob/master/adaptable_dataset_demo.ipynb)


### Recreate

This section provides instructions for recreating the AdapTable dataset.

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

## Reproducibility
Our main experiment setting uses [MetaICL](https://github.com/facebookresearch/MetaICL) for training and testing.

### Model weights
The model weights for our GPT2-large model fine-tuned on `AdapTable-5k` can be downloaded [here](https://drive.google.com/file/d/1Q1mh9rKxD6MX0lTD_okWEjINWRNfqhXY/view?usp=sharing).

### Training
Please follow the instructions on the [MetaICL](https://github.com/facebookresearch/MetaICL) repository for training.

### Evaluation
Please follow the instructions on the [MetaICL](https://github.com/facebookresearch/MetaICL) repository for test evaluation.