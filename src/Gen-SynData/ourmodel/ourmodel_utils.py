import typing as tp

import pandas as pd
import torch
import json

from transformers import AutoTokenizer


def _convert_tokens_to_text(tokens: tp.List[torch.Tensor], tokenizer: AutoTokenizer) -> tp.List[str]:
    """ Decodes the tokens back to strings

    Args:
        tokens: List of tokens to decode
        tokenizer: Tokenizer used for decoding

    Returns:
        List of decoded strings
    """
    # Convert tokens to text
    text_data = [tokenizer.decode(t) for t in tokens]

    # Clean text
    text_data = [d.replace("<|endoftext|>", "") for d in text_data]
    text_data = [d.replace("\n", " ") for d in text_data]
    text_data = [d.replace("\r", "") for d in text_data]

    return text_data


def _convert_text_to_tabular_data(text: tp.List[str], df_gen: pd.DataFrame) -> pd.DataFrame:
    """ Converts the sentences back to tabular data

    Args:
        text: List of the tabular data in text form
        df_gen: Pandas DataFrame where the tabular data is appended

    Returns:
        Pandas DataFrame with the tabular data from the text appended
    """
    columns = df_gen.columns.to_list()
    result_list = []
    extracted_dict = json.loads(text)
    # Convert text to tabular data
    td = dict.fromkeys(columns)
    for col_name, col_value in extracted_dict.items():
        # features = t.split(",")

        # Transform all features back to tabular data
        if col_name in columns and not td[col_name]:
            try:
                td[col_name] = [col_value]
            except IndexError:
                # print("An Index Error occurred - if this happends a lot, consider fine-tuning your model further.")
                pass
    result_list.append(pd.DataFrame(td))
        # df_gen = pd.concat([df_gen, pd.DataFrame(td)], ignore_index=True, axis=0)
    generated_df = pd.concat(result_list, ignore_index=True, axis=0)
    df_gen = pd.concat([df_gen, generated_df], ignore_index=True, axis=0)
    return df_gen
