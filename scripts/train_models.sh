#!/bin/bash
#SBATCH --job-name=example
#SBATCH --mem=20G
#SBATCH --time=99:0:0
#SBATCH -p gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=6
#source ~/anaconda-python
#source ~/.bashrc
#module load gcc/4.9.2

model=conttime
dataname=hawkes
maxepoch=100
dimlstm=32
trainratio=1.0
sizebatch=50
trackperiod=100
prune=0
seed=12345

. parse_options.sh

echo $model
echo $dataname
echo $maxepoch
echo $dimlstm
echo $trainratio
echo $sizebatch
echo $trackperiod
echo $prune
echo $seed


jobname=model$model.data$dataname.me$maxepoch.d$dimlstm.tr$trainratio.sb$sizebatch.tp$trackperiod.prune$prune.sd$seed
echo $jobname

module load cuda/7.5
module load cudnn/5.0
SCRATCHPATH='/scratch/users/hmei2@jhu.edu'
export M_ID=$((${SLURM_ARRAY_TASK_ID}-1))
export THEANO_FLAGS="device=gpu"
python train_models.py -fd ./data/data_$dataname/ -m $model -me $maxepoch -d $dimlstm -tr $trainratio -mt 1 -md 10 -sb $sizebatch -tp $trackperiod -ps $prune -s $seed > $jobname.out
