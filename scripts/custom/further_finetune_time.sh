#!/bin/bash
# Environment Variables
WORLD_SIZE=1
NPROC_PER_NODE=8
MASTER_ADDR="127.0.0.1"
MASTER_PORT=17777
RANK=0

# Training Arguments
GLOBAL_BATCH_SIZE=128
LOCAL_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

# Log Arguments
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=videollama2_vllava
RUN_NAME=videollama2_test
DATA_DIR=video_process
OUTP_DIR=checkpoints

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    videollama2/train_flash_attn.py \
    --deepspeed scripts/zero3_offload.json \
    --version v1_mistral \
    --vision_tower ./openai/clip-vit-large-patch14-336 \
    --mm_projector_type stc_connector \
    --pretrain_mm_mlp_adapter ./checkpoints/VideoLLaMA2-7B-Base/mm_projector.bin \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 \
    --pretrain_model_name_or_path ./checkpoints/VideoLLaMA2-7B \
    --data_path   ${DATA_DIR}/final/pure_normalized_time_conversation_bddx_train.json \
    --data_folder ${DATA_DIR}/BDDX_Processed/ \
    --freeze_backbone True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --mm_use_time_token True \
    --tune_time_token True \
    --num_frames 8 \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --output_dir ${OUTP_DIR}/${WANDB_PROJECT}/further_finetune_normalized_time_${RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size $LOCAL_BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to tensorboard \
    --run_name $RUN_NAME \
