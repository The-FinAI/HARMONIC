# SynData

A project for synthesizing data tables based on a large model.

## Contents

- [SynData](#syndata)
  - [Contents](#contents)
    - [Results](#results)
    - [Examples](#examples)


### Results

<center>Table1. The values of machine learning efficiency(MLE) and large language model efficiency(LLE)</center>

<table>

  <tr>
  <td>Dataset</td>
  <td>Metric</td>
  <td>Original</td>
  <td>OurModel</td>
  <td>Smote</td>
  <td>TVAE</td>
  <td>CTAB</td>
  <td>TabDDPM</td>
  <td>TABSYN</td>
  <td>GReaT</td>
  <td>REaLTabFormer</td>
  




  </tr>

  <tr>
    <td rowspan="2">GM</td>
    <td>MLE</td>
    <td>0.50<sub>±0.00</sub></td>
    <td></td>
    <td>0.64<sub>±0.02</sub></td>
    <td>0.61<sub>±0.02</sub></td>
    <td>0.57<sub>±0.02</sub></td>
    <td>0.64<sub>±0.01</sub></td>
    <td>0.63<sub>±0.02</sub></td>
    <td></td>
    <td>0.65<sub>±0.01</sub></td>
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
    <td></td>
  </tr>

  <tr>
    <td rowspan="2">AD</td>
    <td>MLE</td>
    <td>0.61<sub>±0.00</sub></td>
    <td></td>
    <td>0.75<sub>±0.00</sub></td>
    <td>0.74<sub>±0.00</sub></td>
    <td>0.73<sub>±0.01</sub></td>
    <td>0.74<sub>±0.00</sub></td>
    <td>0.73<sub>±0.01</sub></td>
    <td></td>
    <td>0.7600<sub>±.0027</sub></td>
    
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
    <td></td>
  </tr>

  <tr>
    <td rowspan="2">DI</td>
    <td>MLE</td>
    <td>0.56<sub>±0.00</sub></td>
    <td></td>
    <td>0.72<sub>±0.03</sub></td>
    <td>0.71<sub>±0.02</sub></td>
    <td>0.67<sub>±0.02</sub></td>
    <td>0.71<sub>±0.02</sub></td>
    <td>0.68<sub>±0.03</sub></td>
    <td></td>
    <td>0.66<sub>±0.03</sub></td>
    
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
    <td></td>

  <tr>
    <td rowspan="2">BU</td>
    <td>MLE</td>
    <td>0.8548<sub>±.0001</sub></td>
    <td></td>
    <td>0.8380<sub>±.0056</sub></td>
    <td><sub>±</sub></td>
    <td><sub>±</sub></td>
    <td>0.8440<sub>±.0057</sub></td>
    <td>0.2619<sub>±.0060</sub></td>
    <td></td>
    <td>0.2617<sub>±.0051</sub></td>
    
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
    <td></td>
  </tr>

  <tr>
    <td rowspan="2">AB</td>
    <td>MLE</td>
    <td>0.42<sub>±0.00</sub></td>
    <td></td>
    <td>0.40<sub>±0.01</sub></td>
    <td>0.22<sub>±0.03</sub></td>
    <td>0.24<sub>±0.01</sub></td>
    <td>0.35<sub>±0.02</sub></td>
    <td>0.33<sub>±0.01</sub></td>
    <td></td>
    <td>0.33<sub>±0.02</sub></td>
    

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
    <td></td>

  <tr>
    <td rowspan="2">CA</td>
    <td>MLE</td>
    <td>0.6674<sub>±.0001</sub></td>
    <td></td>
    <td>0.6395<sub>±.0062</sub></sub></td>
    <td>-4.4512<sub>±.5546</sub></td>
    <td>-5.3544<sub>±.7983</sub></td>
    <td>0.6398<sub>±.0518</sub></td>
    <td>0.6304<sub>±.0057</sub></td>
    <td></td>
    <td>-2.0046<sub>±3.9868</sub></td>
    
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
    <td></td>
</table>


<center>Table2. The values of f mean Distance to Closest Record
(DCR), NewRowSynthesis(NRS) and PPL-diff(PPL)</center>

<table>

  <tr>
  <td>Dataset</td>
  <td>Metric</td>
  <td>OurModel</td>
  <td>Smote</td>
  <td>TVAE</td>
  <td>CTAB</td>
  <td>TabDDPM</td>
  <td>TABSYN</td>
  <td>GReaT</td>
  <td>REaLTabFormer</td>
  
  </tr>

  <tr>
    <td rowspan="3">GM</td>
    <td>NRS</td>
    <td></td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td></td>
    <td> 1.00</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>2.77</td>
    <td>4.09</td>
    <td>5.36</td>
    <td>2.21</td>
    <td>3.98</td>
    <td></td>
    <td>4.60</td>
  </tr>
    <tr>
    <td>PPL</td>
    <td></td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="3">AD</td>
    <td>NRS</td>
    <td></td>
    <td>0.95</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td></td>
    <td>0.9954</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.16</td>
    <td>0.49</td>
    <td>0.82</td>
    <td>0.50</td>
    <td>0.86</td>
    <td></td>
    <td>0.59</td>
  </tr>
    <tr>
    <td>PPL</td>
    <td></td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td></td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="3">DI</td>
    <td>NRS</td>
    <td></td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td></td>
    <td>1.00</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.28</td>
    <td>0.33</td>
    <td>0.72</td>
    <td>0.21</td>
    <td>1.37</td>
    <td></td>
    <td>0.36</td>
  </tr>
    <tr>
    <td>PPL</td>
    <td></td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="3">BU</td>
    <td>NRS</td>
    <td></td>
    <td>0.9329</td>
    <td>0.9996</td>
    <td></td>
    <td>0.9905</td>
    <td>0.9990</td>
    <td></td>
    <td>0.9983</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.1485</td>
    <td>0.6608</td>
    <td></td>
    <td>0.1786</td>
    <td>1.3763</td>
    <td></td>
    <td>0.3759</td>
  </tr>
    <tr>
    <td>PPL</td>
    <td></td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="3">AB</td>
    <td>NRS</td>
    <td></td>
    <td> 0.88</td>
    <td> 1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td></td>
    <td>1.00</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.05</td>
    <td>0.18</td>
    <td>0.33</td>
    <td>0.14</td>
    <td>0.13</td>
    <td></td>
    <td>0.11</td>
  </tr>
    <tr>
    <td>PPL</td>
    <td></td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td></td>
    <td></td>
  </tr>

  <tr>
    <td rowspan="3">CA</td>
    <td>NRS</td>
    <td></td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td></td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td></td>
    <td>1.0000</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.0583</td>
    <td>0.1201</td>
    <td>0.1622</td>
    <td>0.0777</td>
    <td>0.1132</td>
    <td></td>
    <td>0.1011</td>
  </tr>
    <tr>
    <td>PPL</td>
    <td></td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td></td>
    <td></td>
  </tr>
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
