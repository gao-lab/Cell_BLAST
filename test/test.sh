#!/bin/bash

set -e

# Python
coverage run --source="../Cell_BLAST" --omit="*/metrics.py" ./test.py
coverage report -m

# R
# ./coverage.R -s ../Utilities/data.R -t ./test.R
