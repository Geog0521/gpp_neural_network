#!/bin/bash
#SBATCH -J neural_network_gpp #it donotes the alias of the shell script
#SBATCH --partition=general-gpu
#SBATCH --account=wer22004
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --constraint="a100"
#SBATCH -o neural_network_gpp_%A_%a.out
#SBATCH -e neural_network_gpp_%A_%a.err
#SBATCH --mail-type=END                              # Event(s) that triggers email notification (BEGIN,END,FAIL,ALL)
#SBATCH --mail-user=xinghua.cheng@uconn.edu          # Destination email address

module purge
module load gcc/11.3.0 gsl/2.7 zlib/1.2.12 libffi/3.2.1
module load gdal/3.6.0

##remove previous files
#rm -r cumula_prec_.*

##rm -r *.out
##rm -r *.err
##rm -r *.log

source /home/xic23015/miniconda3/etc/profile.d/conda.sh
conda activate neural_network_torch   # replace with your own conda environment

cd /home/xic23015/gpp_exercise/

python3 step_3_train_deep_network.py
