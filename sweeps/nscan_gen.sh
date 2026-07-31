#!/bin/bash
# Generate the distance-6 random-regular QA pools for the N-scan (nscan_sweep.sh).
# Pool size = maxB(in the sweep grid) + 64 valid.  seeds 1-4.  ~1.5s per file.
cd "$(dirname "$0")/.."
source ~/miniconda3/etc/profile.d/conda.sh && conda activate conet
set -e
python conet/gen_randreg_frust.py -N 120 -B 384  -D 6 --seeds 1 2 3 4
python conet/gen_randreg_frust.py -N 240 -B 704  -D 6 --seeds 1 2 3 4
python conet/gen_randreg_frust.py -N 480 -B 1344 -D 6 --seeds 1 2 3 4
python conet/gen_randreg_frust.py -N 960 -B 1600 -D 6 --seeds 1 2 3 4
echo "nscan QA pools generated."
