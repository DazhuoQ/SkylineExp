#!/bin/bash

# Assign input arguments to variables
config_file=$1
output_dir=$2

# Ensure the output directory exists
mkdir -p $output_dir

# Define the log file path
log_file="$output_dir/output.log"

# Run the experiment using experiment.py
python -m experiment $config_file $output_dir | tee $log_file