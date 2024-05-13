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
  <td>Smote</td>
  <td>CTGAN</td>
  <td>CTAB</td>
  <td>TabDDPM</td>
  <td>TABSYN</td>
  <td>RTF</td>
  <td>OM</td>




  </tr>

  <tr>
    <td rowspan="2">German</td>
    <td>MLE</td>
    <td></td>
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
  </tr>
  <!-- <tr>
    <td>Named Entity Recognition</td>
    <td>It aims to extract nouns and phrases with legal characteristics from various legal documents.</td>
  </tr>
  <tr>
    <td>Judicial Summarization</td>
    <td>It aims to condense, summarize, and synthesize the content of legal documents.</td>
  </tr>
  <tr>
    <td>Case Recognition</td>
    <td>It aims to determine, based on the relevant description of the case, whether it pertains to a criminal or civil matter.</td>
  </tr> -->

  <tr>
    <td rowspan="2">Adult Income</td>
    <td>MLE</td>
    <td>0.7486±.0022</td>
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
  </tr>
  <!-- <tr>
    <td>Criminal Judgment Prediction</td>
    <td>It involves predicting the guilt or innocence of the defendant, along with the potential sentencing, based on the results of basic legal NLP, including the facts of the case, the evidence presented, and the applicable law articles. Therefore, it is divided into two types of tasks: Charge Prediction and prison Term Prediction.</td>
  </tr>
  <tr>
    <td>Civil Trial Prediction</td>
    <td>It involves using factual descriptions to predict the judgment of the defendant in response to the plaintiff’s claim, which we should consider the Controversial Focus.</td>
  </tr>
  <tr>
    <td>Legal Question Answering</td>
    <td>It utilizes the model’s legal knowledge to address the national judicial examination, which encompasses various specific legal types.</td>
  </tr> -->

  <tr>
    <td rowspan="2">Diabetes</td>
    <td>MLE</td>
    <td></td>
  </tr>
  <tr>
    <td>LLE</td>
    <td></td>
  </tr>
  <!-- <tr>
    <td>Legal Consultation</td>
    <td>It covers a wide range of legal areas and aims to provide accurate, clear, and reliable answers based on the legal questions provided by the different users. Therefore, it usually requires the sum of the aforementioned capabilities to provide professional and reliable analysis.</td>
  </tr> -->

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
