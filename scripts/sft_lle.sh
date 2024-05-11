#! /bin/bash
## sft
export CUDA_VISIBLE_DEVICES=0
model_name_or_path=/media/data1/fengduanyu/llama-2-7b-chat-T/
data_name=diabetes  # german adult diabetes buddy abalone california
seed_gen=2416
############# 以下内容无需修改
syn_method=ourmodel
epoch_num=7
batch_size=32
lr=1e-4
seed=1234
SAVE_PATH=results/FT-CP/llama2-7b-chat-ds/${data_name}-${seed_gen}
mkdir -p $SAVE_PATH

# LoRA without 8bit
torchrun --nproc_per_node 1 --master-port=10086 \
    src/FT/entry_point/sft_train.py \
    --model_name_or_path ${model_name_or_path} \
    --fp16 \
    --use_lora True \
    --llama True \
    --deepspeed src/FT/configs/deepspeed_config_stage3.json \
    --lora_config src/FT/configs/lora_config_llama.json \
    --train_file Data/${data_name}/ft/${syn_method}-${seed_gen}_ft.json \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --num_train_epochs ${epoch_num} \
    --model_max_length 2048 \
    --save_strategy "steps" \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.00001 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --evaluation_strategy "steps" \
    --seed ${seed} \
    --cache_dir ${SAVE_PATH}/hf_cache_dir \
    --output_dir ${SAVE_PATH} \
    --overwrite_output_dir \
    --do_train True


## merge_lora
lora_path=$(python src/FT/get_last_checkpoint.py --folder $SAVE_PATH)
output_path=results/FT-LLMs/llama2-7b-chat-ds/${data_name}-${seed_gen}-ds

python src/FT/merge_llama_with_lora.py \
  --model_name_or_path ${model_name_or_path} \
  --output_path ${output_path} \
  --lora_path ${lora_path} \
  --llama


