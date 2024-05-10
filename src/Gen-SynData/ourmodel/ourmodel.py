import os
import json
import typing as tp

import numpy as np
import pandas as pd

from tqdm import tqdm
import torch
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoConfig)
from ourmodel.ourmodel_utils import _convert_tokens_to_text, _convert_text_to_tabular_data


class OurModel():
    """ OurModel Class

    The OurModel class is used to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(self, llm: str, data: pd.DataFrame, categorical_columns: list = [],):
        """ Initializes Tabula.

        Args:
            llm: a generator(FT LLM), used to generate synthetic data
            data_path: real data path
        """
        # Load Model and Tokenizer from HuggingFace
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.config = AutoConfig.from_pretrained(self.llm)
        # self.model = AutoModelForCausalLM.from_config(self.config)
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        self.data = data
        self.columns = self.data.columns.to_list()

        # Set the training hyperparameters
        # self.categorical_columns = categorical_columns

        # Needed for the sampling process
        # self.columns = None
        # self.num_cols = None
        # self.conditional_col = None
        # self.conditional_col_dist = None

    def tabula_sample(self, starting_prompts: tp.Union[str, list[str]], temperature: float = 0.7, max_length: int = 100,
                      device: str = "cuda", seed: int = 2416) -> pd.DataFrame:
        """ Generate synthetic tabular data samples conditioned on a given input.

        Args:
            starting_prompts: String or List of Strings on which the output is conditioned.
             For example, "Sex is female, Age is 26"
            temperature: The generation samples each token from the probability distribution given by a softmax
             function. The temperature parameter controls the softmax function. A low temperature makes it sharper
             (0 equals greedy search), a high temperature brings more diversity but also uncertainty into the output.
             See this blog article (https://huggingface.co/blog/how-to-generate) to read more about the generation
             process.
            max_length: Maximal number of tokens to generate - has to be long enough to not cut any information
            device: Set to "cpu" if the GPU should not be used. You can also specify the concrete GPU.

         Returns:
            Pandas DataFrame with synthetic data generated based on starting_prompts
        """
        # ToDo: Add n_samples argument to generate more samples for one conditional input.
        self.model.to(device)
        # self.columns = ['CheckingStatus', 'Duration', 'CreditHist', 'Purpose', 'CreditAmt', 'Savings', 'EmploySince',
        #                 'InstallRate', 'PersStatus', 'Debtors', 'ResidSince', 'Property', 'Age', 'InstallPlans',
        #                 'Housing', 'BankCredits', 'Job', 'MaintLiable', 'Telephone', 'ForeignWorker', 'Status']
        starting_prompts = [starting_prompts] if isinstance(starting_prompts, str) else starting_prompts
        generated_data = []

        # Generate a sample for each starting point
        for prompt in tqdm(starting_prompts):
            start_token = torch.tensor(self.tokenizer(prompt)["input_ids"]).to(device)

            # 设置随机种子
            torch.manual_seed(seed)

            # Generate tokens
            gen = self.model.generate(input_ids=torch.unsqueeze(start_token, 0), max_length=max_length,
                                      do_sample=True, temperature=temperature, pad_token_id=50256)
            generated_data.append(torch.squeeze(gen))

        # Convert Text back to Tabular Data
        decoded_data = _convert_tokens_to_text(generated_data, self.tokenizer)
        # df_gen = _convert_text_to_tabular_data(decoded_data, pd.DataFrame(columns=self.columns))
        # 找到"Generate one sample: "后面的位置
        start_index = decoded_data[0].find("Generate one sample: ") + len("Generate one sample: ") + 1
        # 找到下一个句号的位置
        end_index = decoded_data[0].find("}", start_index)
        # 截取字符串
        result = [decoded_data[0][start_index:end_index+1]]
        # import pdb
        # pdb.set_trace()
        df_gen = _convert_text_to_tabular_data(result[0], pd.DataFrame(columns=self.columns))
        return df_gen

    # def load_finetuned_model(self, path: str):
    #     """ Load fine-tuned model
    #
    #     Load the weights of a fine-tuned large language model into the Tabula pipeline
    #
    #     Args:
    #         path: Path to the fine-tuned model
    #     """
    #     self.model.load_state_dict(torch.load(path))

    # def _update_column_information(self, df: pd.DataFrame):
    #     # Update the column names (and numerical columns for some sanity checks after sampling)
    #     self.columns = df.columns.to_list()
    #     self.num_cols = df.select_dtypes(include=np.number).columns.to_list()

    # def _update_conditional_information(self, df: pd.DataFrame, conditional_col: tp.Optional[str] = None):
    #     assert conditional_col is None or isinstance(conditional_col, str), \
    #         f"The column name has to be a string and not {type(conditional_col)}"
    #     assert conditional_col is None or conditional_col in df.columns, \
    #         f"The column name {conditional_col} is not in the feature names of the given dataset"
    #
    #     # Take the distribution of the conditional column for a starting point in the generation process
    #     self.conditional_col = conditional_col if conditional_col else df.columns[-1]
    #     self.conditional_col_dist = _get_column_distribution(df, self.conditional_col)
