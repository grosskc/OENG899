#!/bin/bash

# Open an ipython console connected to the most recently
# launched jupyter kernel

JUPYTER_DIR=`jupyter --runtime-dir`
KERNEL=`ls -t ${JUPYTER_DIR}/kernel-*.json | head -n 1`
jupyter console --existing ${KERNEL}