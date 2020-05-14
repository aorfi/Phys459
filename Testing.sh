#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=00:01:00
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
module load python/3.6
python Testing.py