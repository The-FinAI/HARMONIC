#! /bin/bash
export CUDA_VISIBLE_DEVICES=0
model_name_or_path=Base-Models/llama-2-7b-chat-T
lora_path=results/FT-CP/llama2-7b-chat/german/checkpoint-560
output_path=results/FT-LLMs/German-OurModel-LLaMA2-Chat

python src/FT/merge_llama_with_lora.py \
  --model_name_or_path ${model_name_or_path} \
  --output_path ${output_path} \
  --lora_path ${lora_path} \
  --llama


