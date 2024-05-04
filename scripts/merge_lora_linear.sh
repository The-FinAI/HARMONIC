#! /bin/bash
# Llama2-chat： Base-Models/llama-2-7b-chat-T
# GPT2: Base-Models/ditilgpt2

for epoch_num in 7
do
    model_name_or_path=Base-Models/llama-2-7b-chat-T
    lora_path=/share/fengduanyu/SynData/results/FT-CP/German-OurModel-LLaMA2-Chat_e${epoch_num}_b10_dict/checkpoint-560
    output_path=/share/fengduanyu/SynData/results/FT-LLMs/German-OurModel-LLaMA2-Chat_e${epoch_num}_b10_dict

    CUDA_VISIBLE_DEVICES=7 python src/FT/merge_llama_with_lora.py \
        --model_name_or_path ${model_name_or_path} \
        --output_path ${output_path} \
        --lora_path ${lora_path} \
        --llama
done

#--llama