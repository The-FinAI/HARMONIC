# SynData

A project for synthesizing data tables based on a large model.

## Contents

- [SynData](#syndata)
  - [Contents](#contents)
    - [Examples](#examples)


### Examples

<u>Preprocess Data.</u>

Template and example

```bash
python scripts/preprocess_data.py [data_name] [seed] [knn_n] [task_type] [des] [re_format] [sample_num]
python scripts/preprocess_data.py german 416 5 "binary classification" "user credit scores" dict 700
```

<u>Train Generator.</u>

Template and example

```bash
sh scripts/sft_gen.sh
```

<u>Sample.</u>

Template and example

```bash
python scripts/sample.py [data_name] [sample_num] [seed] [temperature] [max_length] [task_type]
python scripts/sample.py german 700 2416 0.7 2048 'binary classification'
```

<u>Eval</u>

Template and example

```bash
sh scripts/sft_lle.sh
sh scripts/eval-llama2.sh
```
