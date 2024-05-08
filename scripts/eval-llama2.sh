export CUDA_VISIBLE_DEVICES=0
tasks_name=german_train,german_test
pretrained_model=results/FT-LLMs/German-LLaMA2-Chat_e8_b10
write_out_path=results/evalllm_result2/Real-German-Llama2-chat_e8_b10

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