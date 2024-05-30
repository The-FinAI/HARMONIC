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
    <td>0.44<sub>±0.03</td>
    <td>0.65<sub>±0.01</sub></td>
  </tr>
  <tr>
    <td>LLE</td>
    <td>0.7050</td>
    <td>0.70<sub>±0.03</sub></td>
    <td>0.67<sub>±0.03</sub></td>
    <td>0.69<sub>±0.03</sub></td>
    <td>0.71<sub>±0.02</sub></td>
    <td>0.67<sub>±0.05</sub></td>
    <td>0.72<sub>±0.02</sub></td>
    <td>0.55<sub>±0.11</sub></td>
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
    <td>0.73<sub>±0.01</sub></td>
    <td>0.7600<sub>±.0027</sub></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td>0.8077</td>
    <td></td>
    <td>0.84<sub>±0.01</sub></td>
    <td>0.83<sub>±0.01</sub></td>
    <td>0.83<sub>±0.00</sub></td>
    <td>0.83<sub>±0.00</sub></td>
    <td>0.81<sub>±0.02</sub></td>
    <td>0.82<sub>±0.02</sub></td>
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
    <td>0.45<sub>±0.03</sub></td>
    <td>0.66<sub>±0.03</sub></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td>0.6954</td>
    <td>0.75<sub>±0.00</td>
    <td>0.69<sub>±0.04</td>
    <td>0.72<sub>±0.04</td>
    <td>0.62<sub>±0.09</td>
    <td>0.72<sub>±0.03</td>
    <td>0.77<sub>±0.01</td>
    <td>0.71<sub>±0.03</td>
    <td></td>

  <tr>
    <td rowspan="2">BU</td>
    <td>MLE</td>
    <td>0.38<sub>±0.00</sub></td>
    <td></td>
    <td>0.25<sub>±0.02</sub></td>
    <td>0.27<sub>±0.03</sub></td>
    <td>0.26<sub>±0.01</sub></td>
    <td>0.27<sub>±0.01</sub></td>
    <td>0.26<sub>±0.01</sub></td>
    <td>0.24<sub>±0.03</sub></td>
    <td>0.2617<sub>±.0051</sub></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td>0.8804</td>
    <td>0.84<sub>±0.00</sub></td>
    <td>0.85<sub>±0.04</sub></td>
    <td>0.86<sub>±0.01</sub></td>
    <td>0.82<sub>±0.02</sub></td>
    <td>0.85<sub>±0.01</sub></td>
    <td>0.86<sub>±0.01</sub></td>
    <td>0.81<sub>±0.03</sub></td>
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
    <td>0.1017</td>
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
    <td>0.62<sub>±0.00</sub></td>
    <td></td>
    <td>-2.73<sub>±.4.74</sub></sub></td>
    <td>-4.01<sub>±0.11</sub></td>
    <td>-5.02<sub>±0.58</sub></td>
    <td>0.63<sub>±0.00</sub></td>
    <td>0.60<sub>±0.01</sub></td>
    <td></td>
    <td>-2.0046<sub>±3.9868</sub></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td>0.6309</td>
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
    <td>1.00</td>
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
    <td>5.84</td>
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
    <td>1.00</td>
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
    <td>1.51</td>
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
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td>0.43</td>
    <td>0.28</td>
    <td>0.33</td>
    <td>0.72</td>
    <td>0.21</td>
    <td>1.37</td>
    <td>1.36</td>
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
    <td>0.93</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>0.99</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>0.9983</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.15</td>
    <td>0.66</td>
    <td>0.70</td>
    <td>0.18</td>
    <td>1.38</td>
    <td>8.30</td>
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
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td>1.00</td>
    <td></td>
    <td>1.0000</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.06</td>
    <td>0.12</td>
    <td>0.16</td>
    <td>0.08</td>
    <td>0.11</td>
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
