#!/bin/bash
MODEL_PATH='./checkpoints/videollama2_vllava/finetune_normalized_time_videollama2_test'
CUDA_VISIBLE_DEVICES=1 python -m videollama2.serve.eval_custom_predsig --model-path ${MODEL_PATH} --input "./video_process/final/normalized_time_conversation_bddx_eval.json"  --output "results/finetune_normalized_time_videollama2_test" 