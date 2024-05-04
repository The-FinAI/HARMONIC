# llama2-chat: "Base-Models/llama-2-7b-chat-T"
# german: german_train,german_test
# fraud: fraud_train,fraud_test

for data_name in om_e3_b10_t0.7_fil om_e5_b10_t0.7_fil om_e7_b10_t0.7_fil
do
    gpu_devices=7
    #tasks_name=german_train
    tasks_name=german_train,german_test
    model_name=German-LLaMA2-Chat_${data_name}
    pretrained_model=/share/fengduanyu/SynData/results/FT-LLMs/German-LLaMA2-Chat_${data_name}
    write_out_path=/share/fengduanyu/SynData/results/evalllm_result2/$model_name

    log_path=$write_out_path/$tasks_name.log
    mkdir -p "$(dirname "$write_out_path")"
    mkdir -p "$(dirname "$log_path")"
    echo "$log_path"

    export CUDA_VISIBLE_DEVICES="$gpu_devices"
    python src/Eval-Metrics/eval-llm/eval.py \
        --model "hf-causal-llama" \
        --model_args "use_accelerate=True,pretrained=$pretrained_model,tokenizer=$pretrained_model,use_fast=True" \
        --tasks "$tasks_name" \
        --no_cache \
        --write_out \
        --output_base_path "$write_out_path" \
        > "$log_path"
done