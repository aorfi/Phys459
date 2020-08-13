#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=23:00:00
#SBATCH --ntasks=50
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=256M
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=alev.orfi@mail.mcgill.ca

python NetKetCluster.py



