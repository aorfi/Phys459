#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=22:00:00
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=1
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out
#SBATCH --mem-per-cpu=256M
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=alev.orfi@mail.mcgill.ca

python NetKetComparisonCluster.py



