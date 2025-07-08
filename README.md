# Quantum Market Making Agents: DDPG, PQC, and VQC

This repository implements classical and quantum-enhanced market making agents based on Deep Deterministic Policy Gradient (DDPG) and quantum circuit extensions within the MBT-Gym environment.

## Project Overview

This project evaluates:

- DDPG Market Maker Agent: Classical baseline agent
- PQC Market Maker Agent: Parameterised Quantum Circuit agent with fixed gates
- VQC Market Maker Agent: Variational Quantum Circuit agent with trainable gates

All agents were tested within the MBT-Gym model-based trading environment to benchmark performance and policy behaviours.

## Environment Setup

Clone repository:

git clone https://github.com/your-anonymous-repo/quantum-market-making.git
cd quantum-market-making

Create virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r experiment_requirements.txt

## Running the Agents Locally

Example command:

python main.py \
    --agent vqc \
    --episodes 10000 \
    --batch_size 256 \
    --save_path outputs/vqc_model \
    --metrics_path outputs/vqc_metrics.csv

## Running on HPC

To run on a high-performance computing cluster using Slurm:

1. Modify the provided .sbatch file (sbatch/train_agent.sbatch) to set:

- Your environment/module loads
- The correct script paths

2. Submit the job with:

sbatch sbatch/train_agent.sbatch

## Evaluation

Evaluation scripts (e.g. evaluate.py) can be used to compute final policy performance metrics and action distributions after training completes. Example:

python evaluate.py

## Citation

This project builds upon MBT-Gym by Jerome Dockès:

Dockès, J. (2020). MBT-Gym: A Model-Based Trading Environment. GitHub repository. https://github.com/jeromedockes/mbt-gym

The MBT-Gym folder was downloaded from the official GitHub repository and integrated into this project to implement and test the classical and quantum market making agents described above.

## Notes

- This repository is anonymised for blind review.
- No institutional or personal information is included.
- For any queries upon unblinding, please refer to the contact details provided in the final submission.

End of README
