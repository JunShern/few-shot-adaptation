#!/bin/bash

for i in $(seq -f "%02g" 0 50)
do
    sbatch download_and_process_slice.sbatch $i
done