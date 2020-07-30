#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=3:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=alev.orfi@mail.mcgill.ca

python NetKetImport.py



