# default
MODEL_NAME=convnext_base_in22k
EMBEDDING_SIZE=1024
MODEL_PATH=${1:-$MODEL_PATH}
DEVICE_ID=0
EMBEDDING_SIZE=1024
ROOT_PATH=/mnt/work/deepfake_detector/GenImage
FAKE_ROOT_PATH=""
DATASET_NAME=GenImage
SAVE_TXT=../output/results/0123_result.txt
INPUT_SIZE=224
BATCH_SIZE=24
# post_aug_mode=jpeg_40

FAKE_INDEXES=(1 2 3 4 5 6 7 8)  #(1 2 3 4 5 6 7 8)
for FAKE_INDEX in ${FAKE_INDEXES[@]}
do
  echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
  python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
                  --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
                  --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
                  --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE}
done


# do
#   echo FAKE_INDEX:${FAKE_INDEX},MODEL_NAME:${MODEL_NAME},MODEL_PATH:${MODEL_PATH}
#   python train.py --root_path ${ROOT_PATH} --fake_root_path '' --model_name ${MODEL_NAME} \
#                   --input_size ${INPUT_SIZE} --batch_size ${BATCH_SIZE} --device_id ${DEVICE_ID} --is_test \
#                   --model_path ${MODEL_PATH} --is_crop --fake_indexes ${FAKE_INDEX} \
#                   --dataset_name ${DATASET_NAME} --save_txt ${SAVE_TXT} --embedding_size ${EMBEDDING_SIZE} \
#                   --post_aug_mode ${post_aug_mode}
# done

# GenImage_LIST = ['stable_diffusion_v_1_4/imagenet_ai_0419_sdv4', 'stable_diffusion_v_1_5/imagenet_ai_0424_sdv5',
#                  'Midjourney/imagenet_midjourney', 'ADM/imagenet_ai_0508_adm', 'wukong/imagenet_ai_0424_wukong',
#                  'glide/imagenet_glide', 'VQDM/imagenet_ai_0419_vqdm', 'BigGAN/imagenet_ai_0419_biggan']