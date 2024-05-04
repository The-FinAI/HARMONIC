# llama2-chat: "Base-Models/llama-2-7b-chat-T"
# german: german_train,german_test
# fraud: fraud_train,fraud_test

gpu_devices=7
#tasks_name=german_train
tasks_name=german_train,german_test
model_name=Real-German-Llama2-chat_e8_b10
pretrained_model=/share/fengduanyu/SynData/results/TMP_FT_LLMs/German-LLaMA2-Chat_e8_b10
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