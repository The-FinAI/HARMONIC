# SynData

A project for synthesizing data tables based on a large model.

## Contents

- [SynData](#syndata)
  - [Contents](#contents)
    - [Results](#results)
    - [Examples](#examples)


### Results



<table>

  <tr>
  <td>Dataset</td>
  <td>Metric</td>
  <td>Original</td>
  <td>OurModel</td>
  <td>Smote</td>
  <td>CTGAN</td>
  <td>CTAB</td>
  <td>TabDDPM</td>
  <td>TABSYN</td>
  <td>REaLTabFormer</td>
  




  </tr>

  <tr>
    <td rowspan="2">GM</td>
    <td>MLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="2">AD</td>
    <td>MLE</td>
    <td>0.6027<sub>±.0000</sub></td>
    <td></td>
    <td>0.7486<sub>±.0022</sub></td>
    <td></td>
    <td>0.7325<sub>±.0049</sub></td>
    <td></td>
    <td></td>
    <td></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="2">DI</td>
    <td>MLE</td>
    <td>0.5524<sub>±.0068</sub></td>
    <td></td>
    <td>0.7226<sub>±.0255</sub></td>
    <td></td>
    <td>0.6632<sub>±.0317</sub></td>
    <td></td>
    <td></td>
    <td></td>
    

  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>

  <tr>
    <td rowspan="2">BU</td>
    <td>MLE</td>
    <td>0.8548<sub>±.0001</sub></td>
    <td></td>
    <td>0.8380<sub>±.0056</sub></td>
    <td></td>
    <td>0.7934<sub>±.0064</sub></td>
    <td></td>
    <td></td>
    <td></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="2">AB</td>
    <td>MLE</td>
    <td>0.4207<sub>±.0008</sub></td>
    <td></td>
    <td>0.4013<sub>±.0141</sub></td>
    <td></td>
    <td>0.2153<sub>±.0730</sub></td>
    <td></td>
    <td></td>
    <td></td>
    

  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>

  <tr>
    <td rowspan="2">CA</td>
    <td>MLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
    <td></td>
</table>


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
