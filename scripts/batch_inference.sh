#!/bin/bash
sleep $((4 * 60 * 60))
MODEL_PATH='./checkpoints/videollama2_vllava/pretrain_time_mlp_videollama2_vllava'
CUDA_VISIBLE_DEVICES=2 python -m videollama2.serve.eval_custom_predsig --model-path ${MODEL_PATH} --input "./video_process/final/time_conversation_bddx_eval.json"  --output "pretrain_time_mlp_results/" 