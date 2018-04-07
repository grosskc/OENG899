#!/bin/bash

# Open an ipython console connected to the most recently
# launched jupyter kernel

# Using this, I can establish a jupyter kernel from within VSCode,
# execute python code cells (defined with notation "# %%"), and 
# open an ipython console that shares the same kernel. To do this, 
# from a bash terminal within VSCode, I run this script after having
# established a jupyter kernel.

# Location of JSON files containing kernel information
JUPYTER_DIR=`jupyter --runtime-dir`

# Find the latest kernel JSON file by listing them all by time
# and taking the first one
KERNEL=`ls -t ${JUPYTER_DIR}/kernel-*.json | head -n 1`

# open and connect ipython terminal to the same kernel
jupyter console --existing ${KERNEL}