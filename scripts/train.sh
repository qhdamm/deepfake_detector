#!/bin/bash

# Define the paths
real_path="/home/student1/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature/crop"
real_recon_path="/home/student1/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/nature/inpainting"
fake_path="/home/student1/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai/crop"
fake_recon_path="/home/student1/deepfake_detector/DR/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4/train/ai/inpainting"
fake_root_path=f"{real_rcon_path},{fake_path},{fake_recon_path}"

# Other variables
dataset_name="DRCT-2M"
model_name="convnext_base_in22k"
embedding_size=1024
input_size=224
batch_size=64
fake_indexes=2
num_epochs=17
device_id="2,3"
lr=0.0001
is_amp="--is_amp"
is_crop="--is_crop"
num_workers=12
save_flag="_drct_amp_crop"

# Run the python script with the defined parameters
python train_contrastive.py --root_path $real_path \
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
                            --save_flag $save_flag
