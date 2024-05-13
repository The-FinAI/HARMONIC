# SynData

A project for synthesizing data tables based on a large model.

## Contents

- [SynData](#syndata)
  - [Contents](#contents)
    - [Results](#results)
    - [Examples](#examples)


### Results



<table>

<style>
  .number-cell {
    font-size: 14px;
  }
</style>

<style>
  .small-text {
    font-size: 10px; /* 设置小字体的大小 */
  }
</style>

  <tr>
  <td>Dataset</td>
  <td>Metric</td>
  <td>Original</td>
  <td>Smote</td>
  <td>CTGAN</td>
  <td>CTAB</td>
  <td>TabDDPM</td>
  <td>TABSYN</td>
  <td>RTF</td>
  <td>OM</td>




  </tr>

  <tr>
    <td rowspan="2">GM</td>
    <td>MLE</td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
  </tr>
  <tr>
    <td>LLE</td>
        <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
  </tr>

  <tr>
    <td rowspan="2">AD</td>
    <td>MLE</td>
    <td class=number-cell>0.7469</td>
    <td class=number-cell>0.7486<span class="small-text">±.0022</span></td>
    <td class=number-cell></td>
    <td class=number-cell>0.7325<span class="small-text">±.0049</span></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
  </tr>

  <tr>
    <td rowspan="2">DI</td>
    <td>MLE</td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    <td class=number-cell>0.7226<span class="small-text">±.0255</span></td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    <td class=number-cell>0.6632<span class="small-text">±.0317</span></td>
    

  </tr>
  <tr>
    <td>LLE</td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>

  <tr>
    <td rowspan="2">BU</td>
    <td>MLE</td>
    <td class=number-cell></td>
    <td class=number-cell>0.8380<span class="small-text">±.0056</span></td>
    <td class=number-cell></td>
    <td class=number-cell>0.7934<span class="small-text">±.0064</span></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
  </tr>

  <tr>
    <td rowspan="2">AB</td>
    <td>MLE</td>
    <td class=number-cell></td>
    <td class=number-cell>0.4013<span class="small-text">±.0141</span></td>
    <td class=number-cell></td>
    <td class=number-cell>0.2153<span class="small-text">±.0730</span></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    

  </tr>
  <tr>
    <td>LLE</td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>

  <tr>
    <td rowspan="2">CA</td>
    <td>MLE</td>
    <td class=number-cell></td>
    <td class=number-cell>0<span class="small-text">±.0</span></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    
  </tr>
  <tr>
    <td>LLE</td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
    <td class=number-cell></td>
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
