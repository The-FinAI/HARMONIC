# SynData

A project for synthesizing data tables based on a large model.

## Contents

- [SynData](#syndata)
  - [Contents](#contents)
    - [Examples](#examples)


### Examples

<ins>Preprocess Data.</ins>

Template and example

```bash
python scripts/preprocess_data.py [data_name] [seed] [knn_n] [task_type] [des] [re_format] [sample_num]
python scripts/preprocess_data.py german 416 5 "binary classification" "user credit scores" dict 700
```

<ins>Train Generator.</ins>

Template and example

```bash
sh scripts/sft_gen.sh
```

<ins>Sample.</ins>

Template and example

```bash
python scripts/sample.py [data_name] [sample_num] [seed] [temperature] [max_length] [task_type] [device]
python scripts/sample.py german 700 2416 0.7 2048 'binary classification' 'cuda:0'
```

<ins>Eval</ins>

Template and example

```bash
sh scripts/sft_lle.sh
sh scripts/eval-llama2.sh
```
