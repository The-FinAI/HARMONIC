import pandas as pd
import torch
from tqdm import tqdm
import argparse
import random
from torch.nn import CrossEntropyLoss
from sklearn.preprocessing import LabelEncoder


device = torch.device("cuda")


def compute_loss(tokenized_texts, attention_mask, model, tokenizer, add_start_token=False):
    assert tokenized_texts.shape[0] == 1, "The batch_size should be 1, when calculating ppl"
    # reduction="none" --> Do not calculate mean (default: mean )
    loss_func = CrossEntropyLoss(reduction="none")
    # dim=2: (input, target) = ((b, c), (b))
    # dim=3: (input, target) = ((b, c, n), (b, n))
    # b: batch_size, n: sequence_length, c: num_labels (vocab_size)
    if add_start_token:
        tokenizer.bos_token = getattr(tokenizer, "bos_token", "<s>")
        tokenizer.bos_token_id = getattr(tokenizer, "bos_token_id", len(tokenizer) - 101)
        bos_tokens_tensor = torch.tensor(
            [[tokenizer.bos_token_id]] * tokenized_texts.size(dim=0))
        tokenized_texts = torch.cat(
            [bos_tokens_tensor, tokenized_texts], dim=1).to(device)
        attention_mask = torch.cat(
            [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64), attention_mask], dim=1
        ).to(device)
    else:
        tokenized_texts = tokenized_texts.to(device)
        attention_mask = attention_mask.to(device)

    labels = tokenized_texts[:, 1:]

    with torch.no_grad():
        outputs = model(tokenized_texts, attention_mask=attention_mask)
        logits = outputs.logits  # (batch_size, sequence_length, vocab_size)
        logits = logits[:, :-1]  # remove the probability_out of the last token for each sequence
        # first sentence: last token is <bos> and bos_id = 1
        # second sentence: last token is <eos> and eos_id = 2
        loss = loss_func(logits.transpose(1, 2), labels) * attention_mask[:, 1:]
        # (batch_size, sequence_length-1)

    num_tokens = torch.sum(attention_mask).item() - attention_mask.size(0)  # first_token
    return 2 ** (torch.sum(loss).item() / num_tokens)


def load_model_tokenizer():
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).eval()
    model.to(device)
    return model, tokenizer


def main():
    ppl = True
    model, tokenizer = load_model_tokenizer()
    # llama2: special_tokens: 'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '</s>'
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else "[PAD]"
    # os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    data = pd.read_csv(args.dataset)
    input_texts = ppl_data(data)  # 1D-list, val: string

    torch.cuda.empty_cache()

    total_loss = 0
    total_tokens = 0
    for i in tqdm(range(0, len(input_texts), args.batch_size), total=len(input_texts) // args.batch_size):
        start_idx = i
        # The number of the last batch may be less than batch_size
        end_idx = min(i + args.batch_size, len(input_texts))
        batch_texts = input_texts[start_idx:end_idx]
        # add_special_tokens=False --> encode --> input_ids without special_token_id
        # default: padding=False, truncation=False,
        tokenized_texts = tokenizer(batch_texts, add_special_tokens=False, padding=True, truncation=True,
                                    max_length=args.max_tokens, return_tensors="pt")

        # (batch_size, sequence_length (tokens))
        ppl_loss = compute_loss(tokenized_texts=tokenized_texts["input_ids"],
                                attention_mask=tokenized_texts["attention_mask"],
                                model=model,
                                tokenizer=tokenizer,
                                add_start_token=False,
                                )
        total_loss += ppl_loss

    avg_ppl_loss = total_loss / len(input_texts)
    print(f"avg_ppl_loss: {avg_ppl_loss:.4f}\tdata_size: {len(input_texts)}")


def ppl_data(df, col_flag=True):
    ans = []
    if col_flag:
        df.columns = [
            "CheckingStatus", "Duration", "CreditHist", "Purpose", "CreditAmt",
            "Savings", "EmploySince", "InstallRate", "PersStatus", "Debtors",
            "ResidSince", "Property", "Age", "InstallPlans", "Housing",
            "BankCredits", "Job", "MaintLiable", "Telephone", "ForeignWorker", "Status"
        ]
        cat_columns = ["CheckingStatus", "CreditHist", "Purpose", "Savings", "EmploySince", "PersStatus",
                       "Debtors", "Property", "InstallPlans", "Housing", "Job", "Telephone", "ForeignWorker"]
        for column in cat_columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
    col_names = df.columns.tolist()
    for index, row in df.iterrows():
        random.seed(400 + index)
        random.shuffle(col_names)
        tmp = ""
        for col in col_names:
            if col_flag:
                tmp = tmp + f"{col} {row[col]}, "
            else:
                tmp = tmp + f"{col} is {row[col]}, "
        tmp = tmp[:-2] + "."
        ans.append(tmp)
    return ans


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", type=str, default="/share/fengduanyu/SynData/results/FT-LLMs/German-OurModel-LLaMA2-Chat_e3_b10")
    # Data/german/raw/german_train_val.csv
    parser.add_argument("-d", "--dataset", type=str, default="/share/fengduanyu/SynData/Data/german/syn/om_e3_b10_t0.7.csv")
    parser.add_argument("-b", "--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--data-type", type=str, default="csv")

    args = parser.parse_args()

    main()
