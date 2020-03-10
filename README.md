# PhoBert-Sentiment-Classification
Sentiment classification for Vietnamese text using PhoBert

## Overview

This project shows how to finetune the recently released [PhoBERT](https://github.com/VinAIResearch/PhoBERT) for sentiment classification using AIViVN's [comments dataset](https://www.aivivn.com/contests/6).

The model score 0.90849 on the public leaderboard, (winner's solution score 0.90087):
![](https://i.imgur.com/o123cJd.png)

## Model architecture
Here we created a custom classification head on top of the BERT backbone. We concatenated the last 4 hidden representations of the ```[CLS]``` token, which is actually ```<s>``` in this case, and fed it to a simple MLP.

![](https://i.imgur.com/1bYD5dq.png)

## Reproducing the comeptition submission 

### Data preprocessing

Download the competition data from https://www.aivivn.com/contests/6 . Move the ```*.crash``` files to the ```./raw``` folder.

To convert the files to ```.csv``` format, run:

```$python crash2csv.py```

This will create two files ```train.csv``` and ```test.csv``` in your ```./data``` folder.

### Installing VnCoreNLP

Install the pythong bindings:

```$pip3 install  vncorenlp```

Clone the VNCoreNLP repo: https://github.com/vncorenlp/VnCoreNLP

### Downloading PhoBERT 

Follow the instructions in the original repo:

PhoBERT-base:

```
$wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
$tar -xzvf PhoBERT_base_transformers.tar.gz
```

PhoBERT-large:

```
$wget https://public.vinai.io/PhoBERT_large_transformers.tar.gz
$tar -xzvf PhoBERT_large_transformers.tar.gz
```

### Training and testing

To perform training on a single fold, run the following command:

```
python train.py --fold <fold-id> \
--train_path ./data/train.csv \
--dict_path /<path-to-phobert>/dict.txt \
--config_path /<path-to-phobert>/config.json \
--bpe_codes /<path-to-phobert>/bpe.codes \
--pretrained_path /<path-to-phobert>/model.bin \
--ckpt_path ./models
--rdrsegmenter_path /<absolute-path-to>/VnCoreNLP-1.1.1.jar 

```
Note that the ```rdrsegmenter_path``` must be an **absolute** path. To fully reproduce the results, repeat for ```fold-id``` 0 to 4.

To generate the submission file, run the following command, we assume that there are 5 checkpoint named ```model_0.bin``` to ```model_4.bin``` in the ```models``` folder.

```
python infer.py  --test_path ./data/test.csv \
--dict_path /<path-to-phobert>/dict.txt \
--config_path /<path-to-phobert>/config.json \
--bpe_codes /<path-to-phobert>/bpe.codes \
--pretrained_path /<path-to-phobert>/model.bin \
--ckpt_path ./models
--rdrsegmenter_path /<absolute-path-to>/VnCoreNLP-1.1.1.jar 

```

This will generate the submission.csv file in the current folder.
