#!/bin/bash
#SBATCH -A iscrc_altad
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH -t 2:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -J imu_train
#SBATCH -o /leonardo_work/IscrC_ALTAD/logs/%x-%j.out
#SBATCH -e /leonardo_work/IscrC_ALTAD/logs/%x-%j.err

set -euo pipefail

# ---------- 1) Moduli runtime (fondamentali per libstdc++ / CUDA) ----------
module purge
module load cuda/12.2
module load gcc/12.2.0

# ---------- 2) Conda env ----------
source /leonardo_work/IscrC_ALTAD/tools/miniforge3/etc/profile.d/conda.sh
conda activate /leonardo_work/IscrC_ALTAD/envs/alice

# (consigliato) evita oversubscription CPU
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ---------- 3) PointNet++ ops (non installate con pip -> PYTHONPATH) ----------
export PYTHONPATH=/leonardo_work/IscrC_ALTAD/src/Pointnet2_PyTorch/pointnet2_ops_lib:$PYTHONPATH

# (opzionale ma utile) arch A100
export TORCH_CUDA_ARCH_LIST="8.0"

export TORCH_HOME=/leonardo_work/IscrC_ALTAD/.cache/torch
mkdir -p $TORCH_HOME/hub/checkpoints

# ---------- 4) Percorsi: MODIFICA QUI se necessario ----------
PROJECT_DIR=/leonardo_work/IscrC_ALTAD/Radar-Camera_IMU_with_uncertainty
ROOT_DIR=/leonardo_work/IscrC_ALTAD/Alice/Hercules
OUTPUT_ROOT=/leonardo_work/IscrC_ALTAD/Radar-Camera_IMU_with_uncertainty/output_with_uncert

mkdir -p /leonardo_work/IscrC_ALTAD/logs
mkdir -p "$OUTPUT_ROOT"

cd "$PROJECT_DIR"

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH




# ---------- 5) Lancia training ----------
python src/models/imu_seq2seq_lstm_radarstep_twohead_quat_adaloss_rmse_se3patch_inmodelenc_batched.py \
  --mode eval \
  --seqs "00,01,02,03,04,05,06,07,08,09,10,11,12,13,15,16,20" \
  --root_dir "$ROOT_DIR" \
  --pcd_rel Continental/continental_pcd \
  --img_rel stereo_left \
  --gt_rel PR_GT/local_inspva.txt \
  --calib_rel Calibration \
  --output_root "$OUTPUT_ROOT" \
  --wq_init -2.5 \
  --wt_init 0.0 \
  --eval_split test


  

