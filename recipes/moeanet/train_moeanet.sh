#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
: "${MODEL_NAME:?Set MODEL_NAME to a local Llama 3 model directory.}"
: "${DATASET_DIRECTORY:?Set DATASET_DIRECTORY to the split JSON directory.}"
image_directory="${IMAGE_DIRECTORY:-${DATASET_DIRECTORY%/Splits}/Samples}"
output_dir="${OUTPUT_DIR:-${project_dir}/runs/moeanet/First_Stage}"

cd "${project_dir}"

"${python_bin}" "${project_dir}/recipes/moeanet/train_moeanet.py" \
  --model_name "${MODEL_NAME}" \
  --dataset_directory "${DATASET_DIRECTORY}" \
  --image_directory "${image_directory}" \
  --vision_encoder_path "${project_dir}/checkpoints" \
  --output_dir "${output_dir}" \
  --stage_batch_sizes '[32,32,32]' \
  --stage_epochs '[60,50,40]' \
  --gradient_accumulation_steps 1 \
  --warmup_epochs 4 \
  --lr 0.0001 \
  --lora_lr 0.00002 \
  --fluid_loss_weight 1.0 \
  --vision_loss_weight 2.0 \
  --expert_temperature 0.6 \
  --seed 45 \
  --quantization \
  --save_metrics
