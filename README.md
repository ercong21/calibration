# Overview
Codes and Data for the EMNLP 2023 paper "**Unleashing the Multilingual Encoder Potential: Boosting Zero-Shot Performance via Probability Calibration**".

Pretrained multilingual encoder models can directly perform zero-shot multilingual tasks or linguistic probing by reformulating the input examples into cloze-style prompts. This is accomplished by predicting the probabilities of the label words at the masked token position, without requiring any updates to the model parameters. However, the performance of this pattern is limited by the model’s bias toward predicting label words which frequently occurred during the pretraining. These words typically receive high probabilities. To address this is- sue, we combine the models with calibration techniques which modify the probabilities of label words predicted by the models. We first validate the effectiveness of a proposed simple calibration method together with other existing techniques on monolingual encoders in both zero- and few-shot scenarios. We subsequently employ these calibration techniques on multilingual encoders, resulting in obvious performance improvements across a wide range of tasks.

# Content
`data`: machine-translated parallel multilingual dataset of AG News for the evaluation, containing 25 languages.  
`run.sh`: run the calibration experiments.  
`run_zero.py`: run the zero-shot nli-based classification baseline.  

# Run
task: `ag_news` `amazon_polarity` `amazon_star` `xnli` `pawsx` `yahoo` `cola` `mrpc` `qnli` `qqp` `rte` `sst2` `wnli` `amazon_reviews_multi` `ag_news_multi` `pawsx_multi` `xnli_multi`  
model: `bert-base-cased` `roberta-base` `bert-base-multilingual-cased` `xlm-roberta-base`  
calibration: `transform` `penalty` `cbm`  

```
./run.sh [model] [calibration]
```

# Citation
```
@article{nie2023unleashing,
  title={Unleashing the Multilingual Encoder Potential: Boosting Zero-Shot Performance via Probability Calibration},
  author={Nie, Ercong and Schmid, Helmut and Sch{\`u}tze, Hinrich},
  journal={arXiv preprint arXiv:2310.05069},
  year={2023}
}
```
