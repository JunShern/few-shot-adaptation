import argparse
import json
import numpy as np
import pandas as pd
import shutil
import spacy
import tarfile
import uuid

from collections import Counter
from itertools import islice
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List
from urllib.parse import urlparse


pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_colwidth', 60)
pd.set_option('display.width', 1000)

nlp = spacy.load("en_core_web_md")
# !pip install spacytextblob
# !python -m textblob.download_corpora
# !python -m spacy download en_core_web_md
is_valid_pos = {
    "ADJ": True, # adjective
    "ADP": True, # adposition
    "ADV": True, # adverb
    "AUX": True, # auxiliary
    "CCONJ": True, # coordinating conjunction
    "DET": True, # determiner
    "INTJ": True, # interjection
    "NOUN": True, # noun
    "NUM": False, # numeral
    "PART": True, # particle
    "PRON": True, # pronoun
    "PROPN": False, # proper noun
    "PUNCT": False, # punctuation
    "SCONJ": True, # subordinating conjunction
    "SYM": False, # symbol
    "VERB": True, # verb
    "X": False, # other
    "SPACE": False,
}

potential_verbalizer_delimiters = [
    ('[', '] ', ' '),
    # ('[', '] ', '. '),
    # ('[', '] ', ' || '),
    # ('[', '] ', '\t'),
    # ('[', '] ', '\n'),

    # ('', ': ', ' '),
    # ('', ': ', '. '),
    # ('', ': ', ' || '),
    # ('', ': ', '\t'),
    # ('', ': ', '\n'),
    
    # ('|', '| ', ' '),
    # ('|', '| ', '. '),
    # ('|', '| ', ' || '),
    # ('|', '| ', '\t'),
    # ('|', '| ', '\n'),
]

def measure_proseness(text):
    if len(text) == 0:
        return 0
    text = text.lower() # make all lowercase since POS detection excessively classifies uppercase nouns as proper nouns
    doc = nlp(text)
    total_count = 0
    valid_count = 0
    for token in doc:
        if is_valid_pos[token.pos_]:
            valid_count += 1
        total_count += 1
    assert total_count == len(doc)
    return valid_count / len(doc)

def convert_to_df(table, header_row_idx=None):
    assert all([len(row) == len(table[0]) for row in table])

    df = pd.DataFrame(table)
    if header_row_idx is None:
        new_header = [str(idx) for idx in df.columns] # Use column index as column names
    
    else:
        header = [str(name) for name in df.iloc[header_row_idx]] # Grab the first row for the header
        df = df[header_row_idx + 1:] # take the data less the header row

        # Replace any empty column names with the column index
        new_header = []
        for idx, name in enumerate(header):
            if not name: # Column name is empty
                name = str(idx)
            if name in new_header: # Column name is already used
                name = f"{name}_{idx}"
            new_header.append(name)
        
    df.columns = new_header # Apply new header
    return df

def is_mostly_valid_text(df, min_proseness):
    enumerated_sample_rows = islice(df.iterrows(), 1, 4) # Skip the first row (might be a header), take 3 rows
    score = np.mean([measure_proseness(' '.join(row)) for idx, row in enumerated_sample_rows])
    if score >= min_proseness:
        return True
    else:
        return False

def make_taskpairs_from_table(df, output_col_name, max_label_len=30, verbalizer_delimiters=('[', '] ', ' ')):
    assert len(verbalizer_delimiters) == 3, f"Invalid verbalizer delimiters: {verbalizer_delimiters}"
    label_left, label_right, sep = verbalizer_delimiters

    output_col_name = str(output_col_name)
    col_names = [str(el) for el in df.columns.to_list()]
    assert output_col_name in col_names, (col_names, output_col_name)
    # Drop all rows with empty output strings
    df = df[df[output_col_name].astype(str).astype(bool)]

    outputs = df[output_col_name].to_list()

    df = df.drop([output_col_name], axis=1)
    
    task_pairs = []
    for (idx, row), output in zip(df.iterrows(), outputs):
        assert output, output
        # Concatenate column names and cell values; 
        input_items = []
        for col_label, value in zip(row.index, row):
            if not value.strip(): # Skip empty input items 
                continue
            # Skip labels that are too long (there may be mislabeled headers that are actually cell values)
            if len(col_label) < max_label_len:
                input_items.append(f"{label_left}{col_label}{label_right}{value}{sep}")
            else:
                input_items.append(f"{value}{sep}")
        input = ''.join(input_items)
        if len(output_col_name) < max_label_len:
            input += f"{label_left}{output_col_name}{label_right}"
        task_pairs.append((input, output))
    return task_pairs

def measure_class_balance(counter: Counter):
    """
    Shannon entropy-based measure of class balance:
    https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
    """
    n = np.sum([c for c in counter.values()])
    k = len(counter)
    numerator = -np.sum([(c / n) * np.log(c / n) for c in counter.values()])
    return numerator / np.log(k)

def get_payleveldomain(url):
    domain = urlparse(url).netloc
    if domain.startswith('www.'):
        domain = domain[len('www.'):]
    return domain

def sanitize_filename(filename):
    clean = "".join([(c if c.isalnum() else '_') for c in filename]).rstrip()
    return clean

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--tarfile", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--max_source_files", type=int, default=None)
    parser.add_argument("--min_rows", type=int, default=6)
    parser.add_argument("--max_tasks_per_domain", type=int, default=50)
    
    args = parser.parse_args()
    
    export_slice = Path(args.tarfile).stem
    if args.outdir:
        out_dir = Path(args.outdir) / export_slice
    else:
        out_dir = Path(args.tarfile).parent / export_slice
    out_dir.mkdir(parents=True, exist_ok=False)

    index_file = out_dir / "index.txt" # Full list of all tasks for ease of human browsing (not consumed by downstream software)
    assert not index_file.exists()

    assert tarfile.is_tarfile(args.tarfile)

    filter_stage_counts = Counter([
        "tables_initial",
        "tables_rejected_minrows",
        "tables_rejected_proseness",
        "tables_remaining",
        "tasks_initial",
        "tasks_rejected_maxdomain",
        "tasks_rejected_taskminrows",
        "tasks_rejected_onetomany",
        "tasks_rejected_minclasses",
        "tasks_rejected_taskminrows",
        "tasks_rejected_outputproseness",
        "tasks_rejected_classbalance",
        "tasks_remaining",
    ])

    domain_counts = Counter()
    with tarfile.open(args.tarfile, "r") as file:
        for idx, member in tqdm(enumerate(islice(file, args.max_source_files)), total=args.max_source_files):

            f = file.extractfile(member)
            if f is None:
                continue
            content = f.read().decode("utf-8")
            obj = json.loads(content)
            filter_stage_counts['tables_initial'] += 1

            # Load table data
            table = obj['relation']

            # Transpose table
            if obj['tableOrientation'] == 'HORIZONTAL':
                table = list(map(list, zip(*table)))
            
            # Convert to dataframe
            header_row_idx = obj['headerRowIndex'] if obj['hasHeader'] else None
            df = convert_to_df(table, header_row_idx=header_row_idx)
            df = df.drop_duplicates()

            if df.shape[0] < args.min_rows:
                filter_stage_counts['tables_rejected_minrows'] += 1
                continue

            if not is_mostly_valid_text(df, min_proseness=0.8):
                filter_stage_counts['tables_rejected_proseness'] += 1
                continue

            filter_stage_counts['tables_remaining'] += 1

            for output_col_name in df.columns: # Consider each column as a potential output column to create a new task
                filter_stage_counts['tasks_initial'] += 1

                domain = get_payleveldomain(obj['url'])
                if domain_counts[domain] >= args.max_tasks_per_domain:
                    filter_stage_counts['tasks_rejected_maxdomain'] += 1
                    continue
                
                verbalizer_delimiters = potential_verbalizer_delimiters[filter_stage_counts['tasks_remaining'] % len(potential_verbalizer_delimiters)]
                xy_pairs = make_taskpairs_from_table(df, output_col_name=output_col_name, verbalizer_delimiters=verbalizer_delimiters)
                if len(xy_pairs) < args.min_rows:
                    filter_stage_counts['tasks_rejected_taskminrows'] += 1
                    continue
                
                all_inputs = [x for x, y in xy_pairs]
                is_one_to_many_mapping = len(set(all_inputs)) < len(all_inputs) # Given no xy duplicates, any duplicate x implies that x maps to different y's
                if is_one_to_many_mapping:
                    filter_stage_counts['tasks_rejected_onetomany'] += 1
                    continue
                
                output_class_counts = Counter([y for x, y in xy_pairs])
                if len(output_class_counts) <= 1:
                    filter_stage_counts['tasks_rejected_minclasses'] += 1
                    continue
                
                output_space = list(output_class_counts.keys())
                if measure_proseness(' '.join(output_space)) < 0.8:
                    filter_stage_counts['tasks_rejected_outputproseness'] += 1
                    continue

                class_balance_score = measure_class_balance(output_class_counts)
                if class_balance_score < 0.7:
                    filter_stage_counts['tasks_rejected_classbalance'] += 1
                    continue

                filter_stage_counts['tasks_remaining'] += 1
                domain_counts[domain] += 1

                # By default, use all outputs in the column for multiple-choice options, unless there are too many or the labels are too long
                # (if we don't provide options this will be treated as a generative task)
                if len(output_space) > 10 or np.mean([len(label) for label in output_space]) > 20:
                    options = []
                else:
                    options = output_space

                short_uuid = str(uuid.uuid4())[:8] # Truncate to 8 doesn't guarantee no collisions, but after adding the titles we're fine
                task_title = "_".join([obj['pageTitle'][-30:], obj['title'][-30:], output_col_name[-30:]])
                task_unique_name = f"{short_uuid}_{sanitize_filename(task_title)}"
                
                # Save task to file
                outfile = out_dir / f"{(task_unique_name)}.jsonl"
                with open(outfile, 'w') as f:
                    for x, y in xy_pairs:
                        datapoint = {
                            "task": task_unique_name,
                            "input": x,
                            "output": y,
                            "options": options,
                            # Additional metadata
                            "pageTitle": obj['pageTitle'],
                            "title": obj['title'],
                            "outputColName": output_col_name,
                            "url": obj['url'],
                            "wdcFile": member.name,
                        }
                        print(json.dumps(datapoint), file=f)

                # Save all tasks in a single file for quick review
                with open(index_file, 'a') as f:
                    print(json.dumps({
                        "task": task_unique_name,
                        # Additional metadata
                        "pageTitle": obj['pageTitle'],
                        "title": obj['title'],
                        "outputColName": output_col_name,
                        "url": obj['url'],
                        "wdcFile": member.name,
                    }, indent=4), file=f)
                    for x, y in xy_pairs:
                        print(f"INPUT:   {x}", file=f)
                        print(f"OPTIONS: {options}", file=f)
                        print(f"OUTPUT:  {y}", file=f)
                        print("---", file=f)
                    print("", file=f)

    with open(index_file, 'a') as f:
        print(f"{len(domain_counts)=}", file=f)
        print(f"{sum(domain_counts.values())=}", file=f)
        print("", file=f)
        for stage, count in filter_stage_counts.items():
            print(f"{stage}\t{count}", file=f)