export CUDA_VISIBLE_DEVICES=0
data_name=diabetes  # german adult diabetes buddy abalone california
seed_gen=2416
############# 以下内容无需修改
tasks_name=${data_name}_test
pretrained_model=results/FT-LLMs/llama2-7b-chat-ds/${data_name}-${seed_gen}-ds
write_out_path=results/Eval-LLMs/llama2-7b-chat-ds/${data_name}-${seed_gen}

log_path=${write_out_path}/${tasks_name}.log
mkdir -p "$(dirname "${write_out_path}")"
mkdir -p "$(dirname "${log_path}")"
echo "${log_path}"

python src/Eval-Metrics/eval-llm/eval.py \
    --model "hf-causal-llama" \
    --model_args "use_accelerate=True,pretrained=$pretrained_model,tokenizer=$pretrained_model,use_fast=True" \
    --tasks "${tasks_name}" \
    --no_cache \
    --write_out \
    --output_base_path "$write_out_path" \
    > "${log_path}"