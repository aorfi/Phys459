#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out

python Testing.py