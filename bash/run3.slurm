#! /bin/bash
#SBATCH --account=p_masi_gpu
#SBATCH --partition=pascal
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH --output=/scratch/yaoy4/SceneClassification/log/run3.log
setpkgs -a tensorflow_0.12
setpkgs -a anaconda2
source activate FCN
cd /scratch/yaoy4/SceneClassification/
python  /scratch/yaoy4/SceneClassification/main.py exp exp3
