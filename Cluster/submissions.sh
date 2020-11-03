#!/bin/bash
#SBATCH --account=def-coish
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --job-name=test
#SBATCH --output=%x-%j.out

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=alev.orfi@mail.mcgill.ca

python ClusterRunningManyNodesN3.py
python ClusterRunningManyNodesN4.py
python ClusterRunningManyNodesN5.py
python ClusterRunningManyNodesN6.py
python ClusterRunningManyNodesN7.py
python ClusterRunningManyNodesN8.py


