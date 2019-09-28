#!/bin/bash
# properties = {properties}

source /lustre1/gaog_pkuhpc/users/caozj/.env && conda activate cb-gpu

{exec_job}
