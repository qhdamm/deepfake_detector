#!/bin/bash

EXPERIMENT_NAME=$1

if [ -z "$EXPERIMENT_NAME" ]; then
    echo "실험 이름을 지정해주세요. 사용법: ./train.sh <experiment_name>"
    exit 1
fi


# Define the paths
# real_path="/mnt/work/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"  # for DIRE
# fake_root_path="dire"  # for DIRE
real_path="/mnt/work/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature/crop"
real_recon_path="/mnt/work/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature/inpainting"
fake_path="/mnt/work/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai/crop"
fake_recon_path="/mnt/work/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai/inpainting"
fake_root_path="${real_recon_path},${fake_path},${fake_recon_path}"

# Other variables
dataset_name="GenImage"
model_name="f3net"
embedding_size=1024
input_size=299
batch_size=64
fake_indexes=2
num_epochs=17
device_id="0"
lr=0.0001
is_amp="--is_amp"
is_crop="--is_crop"
num_workers=12
loss_mode="drct"
data_size=0  # 0이면 모든 데이터
save_flag="ext_${EXPERIMENT_NAME}"


# Run the python script with the defined parameters
python /mnt/work/deepfake_detector/scripts/train.py --root_path "$real_path" \
                            --fake_root_path $fake_root_path \
                            --dataset_name $dataset_name \
                            --model_name $model_name \
                            --embedding_size $embedding_size \
                            --input_size $input_size \
                            --batch_size $batch_size \
                            --fake_indexes $fake_indexes \
                            --num_epochs $num_epochs \
                            --device_id $device_id \
                            --lr $lr \
                            $is_amp \
                            $is_crop \
                            --num_workers $num_workers \
                            --loss_mode $loss_mode \
                            --data_size $data_size \
                            --save_flag $save_flag 
