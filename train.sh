#!/bin/bash
#SBATCH -o %j.out
#SBATCH -e %j.err
#SBATCH --partition=titanx-long    # Partition to submit to
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=10000   # Memory in MB per cpu allocated


LOG_DIR=${LOG_DIR:-shapenet_vae}
INPUT_DATA=${INPUT_DATA:-shapenetcore_partanno_segmentation_benchmark_v0}
MAIN_FILE=${MAIN_FILE:-train_vae.py}
RESUME=${RESUME:-No}
echo "LogDir $LOG_DIR"
echo "InputData $INPUT_DATA"
echo "MainFile $MAIN_FILE"

if [ $RESUME == "No" ]; then
  args="--inp_data $INPUT_DATA --log_dir $LOG_DIR --train"
else
  args="--inp_data $INPUT_DATA --log_dir $LOG_DIR --train --resume $RESUME"
fi

export PYTHONUNBUFFERED="True"

PY_EXE=/home/maverkiou/miniconda2/envs/ocnn_tf1.14/bin/python
SOURCE_DIR=/home/maverkiou/zavou/pointnet-autoencoder

VERSION=$(git rev-parse HEAD)

TIME=$(date +"%Y-%m-%d_%H-%M-%S")
LOG="$LOG_DIR/$TIME.txt"

mkdir -p $LOG_DIR

echo Logging output to "$LOG"
echo "Version: ${VERSION}" > "$LOG"
echo -e "GPU(s): $CUDA_VISIBLE_DEVICES" >> $LOG
echo "cd ${SOURCE_DIR} && ${PY_EXE} ${MAIN_FILE} $args" >> "$LOG"
cd ${SOURCE_DIR} && ${PY_EXE} ${MAIN_FILE} $args 2>&1 | tee -a "$LOG"
