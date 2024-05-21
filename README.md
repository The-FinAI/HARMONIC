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
    <td>0.4971<sub>±.0032</sub></td>
    <td></td>
    <td>0.6405<sub>±.0159</sub></td>
    <td>0.6035<sub>±.0281</sub></td>
    <td>0.5778<sub>±.0263</sub></td>
    <td>0.6468<sub>±.0162</sub></td>
    <td>0.6256<sub>±.0214</sub></td>
    <td></td>
    <td>0.6358<sub>±.0211</sub></td>
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
    <td>0.6027<sub>±.0000</sub></td>
    <td></td>
    <td>0.7486<sub>±.0022</sub></td>
    <td>0.7381<sub>±.0024</sub></td>
    <td>0.7323<sub>±.0042</sub></td>
    <td>0.7456<sub>±.0050</sub></td>
    <td>0.7358<sub>±.0036</sub></td>
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
    <td>0.5524<sub>±.0068</sub></td>
    <td></td>
    <td>0.7226<sub>±.0255</sub></td>
    <td>0.7028<sub>±.0304</sub></td>
    <td>0.6708<sub>±.0211</sub></td>
    <td>0.7130<sub>±.0191</sub></td>
    <td>0.6837<sub>±.0254</sub></td>
    <td></td>
    <td>0.6786<sub>±.0304</sub></td>
    
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
    <td>0.4207<sub>±.0008</sub></td>
    <td></td>
    <td>0.4013<sub>±.0141</sub></td>
    <td>0.1847<sub>±.0495</sub></td>
    <td>0.2490<sub>±.0148</sub></td>
    <td>0.3652<sub>±.0208</sub></td>
    <td>0.3414<sub>±.0236</sub></td>
    <td></td>
    <td>0.3339<sub>±.0221</sub></td>
    

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
    <td>0.9997</td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td></td>
    <td> 0.9980</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>2.7486</td>
    <td>4.1133</td>
    <td>5.3840</td>
    <td>2.2313</td>
    <td>3.9990</td>
    <td></td>
    <td>4.5721</td>
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
    <td>0.9514</td>
    <td>1.0000</td>
    <td>0.9982</td>
    <td>0.9963</td>
    <td>0.9976</td>
    <td></td>
    <td>0.9954</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.1525</td>
    <td>0.4800</td>
    <td>0.8124</td>
    <td>0.4922</td>
    <td>0.8466</td>
    <td></td>
    <td>0.5980</td>
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
    <td>1.0000</td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td>0.9992</td>
    <td>1.0000</td>
    <td></td>
    <td>0.9988</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.2752</td>
    <td>0.3374</td>
    <td>0.7237</td>
    <td>0.2142</td>
    <td>1.3738</td>
    <td></td>
    <td>0.3500</td>
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
    <td> 0.8834</td>
    <td> 1.0000</td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td>1.0000</td>
    <td></td>
    <td>0.9998</td>
  </tr>
  <tr>
    <td>DCR</td>
    <td></td>
    <td>0.0471</td>
    <td>0.1841</td>
    <td>0.3341</td>
    <td>0.1368</td>
    <td>0.1277</td>
    <td></td>
    <td>0.1080</td>
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
