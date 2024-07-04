#!/bin/bash
MODEL_PATH='./checkpoints/videollama2_vllava/pretrain_time_videollama2_3600'
CUDA_VISIBLE_DEVICES=2 python -m videollama2.serve.eval_custom_predsig --model-path ${MODEL_PATH} --input "./video_process/final/time_conversation_bddx_eval.json"  --output "pretrain_time_videollama2_3600_results/" 