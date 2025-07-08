#!/bin/bash
#SBATCH --job-name=market_maker_train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

# Load modules (adjust based on your HPC)
module purge
module load anaconda3
module load cuda/11.8  # Example; replace with your system's CUDA version if needed

# Activate your virtual environment
source activate .venv  # Replace '.venv' with your environment name

# Navigate to your project directory (adjust path)
cd /path/to/your/project

# Run your Python script with arguments
python train_script.py \
    --agent ddpg \
    --episodes 10000 \
    --batch_size 256 \
    --save_path outputs/ddpg_model \
    --metrics_path outputs/ddpg_metrics.csv
