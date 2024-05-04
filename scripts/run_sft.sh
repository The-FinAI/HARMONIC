# smote ctgan ctab+ tabddpm tabsyn great_gpt2 tabula_gpt2 realtabformer_gpt2 taptap_gpt2
#export WANDB_RUN_ID=${data_name}
# 3e-4
export CUDA_VISIBLE_DEVICES=7
for epoch_num in 5
do
    model_name_or_path=Base-Models/llama-2-7b-chat-T
    export WANDB_PROJECT=German-LLaMA2-Chat
    export WANDB_RUN_ID=e${epoch_num}
    train_file=/share/fengduanyu/SynData/Data/german/ft/german_train_val_ft.json
    dev_file=/share/fengduanyu/SynData/Data/german/ft/german_train_val_ft_dev.json
    export WANDB_MODE=disabled
    SAVE_PATH=results/FT-CP/${WANDB_PROJECT}_${WANDB_RUN_ID}_b64_lr1e-4
    mkdir -p ${SAVE_PATH}
    cache_dir=${SAVE_PATH}/hf_cache_dir
    mkdir -p ${cache_dir}

    # LoRA without 8bit
    torchrun --nproc_per_node 1 src/FT/entry_point/sft_train.py \
        --model_name_or_path ${model_name_or_path} \
        --fp16 \
        --use_lora True \
        --llama True \
        --deepspeed src/FT/configs/deepspeed_config_stage3.json \
        --lora_config src/FT/configs/lora_config_llama.json \
        --train_file ${train_file} \
        --validation_file ${dev_file} \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --gradient_accumulation_steps 1 \
        --num_train_epochs ${epoch_num} \
        --model_max_length 2048 \
        --save_strategy "steps" \
        --save_total_limit 1 \
        --learning_rate 1e-4 \
        --weight_decay 0.00001 \
        --warmup_ratio 0.01 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --evaluation_strategy "steps" \
        --seed 1234 \
        --gradient_checkpointing \
        --cache_dir ${cache_dir} \
        --output_dir ${SAVE_PATH} \
        --overwrite_output_dir \
        --do_train True \
        > ${cache_dir}/train.log
done