import argparse

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

# Ref: https://github.com/tloen/alpaca-lora/blob/main/export_hf_checkpoint.py

def apply_lora(model_name_or_path, output_path, lora_path):
    print(f"Loading the base model from {model_name_or_path}")
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
    )
    if args.llama:
        base_tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path)
    else:
        base_tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    print(f"Loading the LoRA adapter from {lora_path}")

    lora_model = PeftModel.from_pretrained(
        base,
        lora_path,
        torch_dtype=torch.float16,
    )

    print("Applying the LoRA")
    model = lora_model.merge_and_unload()
    
    # 修改生成配置 
    # generation_config = model.config.generation 
    # if not generation_config.do_sample: 
#        print("Warning: 'do_sample' is set to False, adjusting parameters...")
 #       generation_config.do_sample = True 
  #      generation_config.temperature = None 
   #     generation_config.top_p = None
    

    print(f"Saving the target model to {output_path}")
    model.save_pretrained(output_path)
    base_tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=True)
    parser.add_argument("--llama", action="store_true")

    args = parser.parse_args()
    # args.llama = True
    if args.llama:
        print(f"正在使用llama：{args.llama}")
    else:
        print(f"没有使用llama：{args.llama}")
    apply_lora(args.model_name_or_path, args.output_path, args.lora_path)
    