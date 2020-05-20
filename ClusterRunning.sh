#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=00:01:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out

python ClusterRunning.py