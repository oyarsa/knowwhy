# KnowWhy

This repository contains all the relevant code and artifacts for the paper - Using Commonsense Knowledge to Answer Why Questions (Lal et al, EMNLP 2022)

## Setup

Please visit [this Zenodo link](https://zenodo.org/record/7709865#.ZAkw-i-B35g) to download the models, data and other artifacts associated with the paper. Once the zip file has been downloaded and unzipped at the desired location, please follow the instructions below:

```
cd knowwhy/
# Tested with Python 3.10
pip install -r requirements.txt
# Installation below only required to generate automatic metric scores
cd src/
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip .
unzip bleurt-base-128.zip
rm -rf bleurt-base-128.zip
```

## Models

### Settings

- `input_type` is `normal` - Model input format: "question: Q context: C"
- `input_type` is `separator-normal` - Model input format: "question: Q \\n context: C"
For the other supported input types:
- if `input_type` contains `original`, COMET inferences are ranked according to their associated COMET scores and used in that order
- if `input_type` contains `reranked`, COMET inferences are ranked according to a trained reranking model and used in that order
- if `input_type` contains `diverse`, COMET inferences are ranked according to a trained reranking model with a post-hoc diversity constraint and used in that order
- if `input_type` contains a number, that number denotes the number of COMET inferences used in the model input
- if `input_type` contains `tup` - Model input format: "question: Q context: C <info> relation: {rel_type} phrase: {rel_phrase} </info>"
- if `input_type` contains `tupsep` - Model input format: "question: Q \\n context: C \\n <info> relation: {rel_type} phrase: {rel_phrase} </info> \\n"
- if `input_type` contains `verbalized` - Model input format: "question: Q \\n context: C \\n verbalized_relation \\n"

### Training

This section contains the commands used to train T5-base versions of all the models reported in the paper

```
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type normal --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_normal.log --model-dir ../models/t5base_normal
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type separator-normal --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_w_n_separator.log --model-dir ../models/t5base_w_n_separator
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type diverse-tup-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_diverse_tup_3.log --model-dir ../models/t5base_diverse_tup_3
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type diverse-tupsep-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_diverse_tupsep_3.log --model-dir ../models/t5base_diverse_tupsep_3
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type diverse-verbalized-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_diverse_verbalized_3.log --model-dir ../models/t5base_diverse_verbalized_3
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type diverse-verbalized-1 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_diverse_verbalized_1.log --model-dir ../models/t5base_diverse_verbalized_1
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type diverse-verbalized-5 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_diverse_verbalized_5.log --model-dir ../models/t5base_diverse_verbalized_5
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type original-verbalized-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_original_verbalized_3.log --model-dir ../models/t5base_original_verbalized_3
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode train --input-type reranked-verbalized-3 --knowledge-folder ../data/comet_ranked_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/finetune_t5base_reranked_verbalized_3.log --model-dir ../models/t5base_reranked_verbalized_3
```

### Inference

This section contains the commands used to run inference for T5-base versions of all the models reported in the paper. Please note that this is done on the TellMeWhy hidden test set. To run inference on the full test set, just remove the `--hidden-test` argument from the command and modify it to reflect new filenames.

```
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type normal --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_normal.log --model-dir ../models/t5base_normal/model.pt --test-output-file ../artifacts/model_predictions/t5base.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type separator-normal --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_w_n_separator.log --model-dir ../models/t5base_w_n_separator/model.pt --test-output-file ../artifacts/model_predictions/t5base_w_n_separator.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type diverse-tup-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_diverse_tup_3.log --model-dir ../models/t5base_diverse_tup_3/model.pt --test-output-file ../artifacts/model_predictions/t5base_diverse_tup_3.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type diverse-tupsep-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_diverse_tupsep_3.log --model-dir ../models/t5base_diverse_tupsep_3/model.pt --test-output-file ../artifacts/model_predictions/t5base_diverse_tupsep_3.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type diverse-verbalized-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_diverse_verbalized_3.log --model-dir ../models/t5base_diverse_verbalized_3/model.pt --test-output-file ../artifacts/model_predictions/t5base_diverse_verbalized_3.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type diverse-verbalized-1 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_diverse_verbalized_1.log --model-dir ../models/t5base_diverse_verbalized_1/model.pt --test-output-file ../artifacts/model_predictions/t5base_diverse_verbalized_1.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type diverse-verbalized-5 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_diverse_verbalized_5.log --model-dir ../models/t5base_diverse_verbalized_5/model.pt --test-output-file ../artifacts/model_predictions/t5base_diverse_verbalized_5.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type original-verbalized-3 --knowledge-folder ../data/comet_sorted_by_raw_scores_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_original_verbalized_3.log --model-dir ../models/t5base_diverse_original_verbalized_3/model.pt --test-output-file ../artifacts/model_predictions/t5base_original_verbalized_3.csv --hidden-test
CUDA_VISIBLE_DEVICES=0 python t5_finetuning.py --mode test --input-type reranked-verbalized-3 --knowledge-folder ../data/comet_ranked_for_tellmewhy --dataset-name tellmewhy --data-folder ../data/tellmewhy/ --log-file ../logs/hidden_test_t5base_reranked_verbalized_3.log --model-dir ../models/t5base_diverse_reranked_verbalized_3/model.pt --test-output-file ../artifacts/model_predictions/t5base_reranked_verbalized_3.csv --hidden-test
```

## Generating COMET data

This section contains the commands used to generate COMET data for the entire TellMeWhy dataset

```
python comet2020_generation.py --input-file ../data/tellmewhy/train.json --output-file ../data/comet_clean_for_tellmewhy/train.json
python comet2020_generation.py --input-file ../data/tellmewhy/val.json --output-file ../data/comet_clean_for_tellmewhy/val.json
python comet2020_generation.py --input-file ../data/tellmewhy/test_full.json --output-file ../data/comet_clean_for_tellmewhy/test_full.json
python comet2020_generation.py --input-file ../data/tellmewhy/test_annotated.json --output-file ../data/comet_clean_for_tellmewhy/test_annotated.json
```

## Evaluation

This section contains example commands to run automatic evaluation as well as use human evaluation judgments to generate Likert scores reported in the paper for each model.
To see all the human evaluation numbers in one place, please check `notebooks/Human_Eval_Numbers.ipynb`.

**Automatic Metrics** - `python automatic_metrics.py --test-output-file ../data/model_predictions/t5base.csv --log-file ../logs/t5base.auto_metrics --temp-dir ../auto-metrics-store/t5base`

**Human Evaluation using judgment cache** - `python evaluator.py --predictions-file-path ../artifacts/model_predictions/gpt3_w_knowl.csv --answers-file-path ../artifacts/human_eval_cache.pkl --output-file-path gpt3_w_knowl.human_eval.json`

## Files and folders

`data/tellmewhy` - Original TellMeWhy data

`data/comet_clean_for_tellmewhy` - COMET2020 output along axes of desired relations paired with TellMeWhy metas

`data/comet_sorted_by_raw_scores_for_tellmewhy` - COMET2020 output along axes of desired relations sorted by COMET scores itself paired with TellMeWhy metas

`data/comet_ranked_for_tellmewhy` - COMET2020 output along axes of desired relations as ranked by Tanvi's ranker model paired with TellMeWhy metas but without any scores

`src/data.py` - File containing functions to load various types of data and create different model inputs

`src/t5_finetuning.py` - Main file for model training and inference

`src/comet2020_generation.py` - File to use COMET and generate relations and scores for TellMeWhy

`src/automatic_metrics.py` - File used to calculate various automatic metrics using the scheme described in (Lal et. al, 2021) for the full TellMeWhy test set; Usage example: `python automatic_metrics.py --test-output-file ../data/model_predictions/t5base.csv --log-file ../logs/t5base.auto_metrics --temp-dir ../auto-metrics-store/t5base`

`src/evaluator.py` - File used to calculate metrics using human judgments in a cache for model answers to the hidden TellMeWhy test set; Usage example: `python evaluator.py --predictions-file-path ../artifacts/model_predictions/gpt3_w_knowl.csv --answers-file-path ../artifacts/human_eval_cache.pkl --output-file-path gpt3_w_knowl.human_eval.json`

`notebooks/Human_Eval_Numbers.ipynb` - Jupyter notebook with all the human evaluation numbers reported in the paper

`artifacts/model_predictions/` - Folder contains all the predictions made using T5-base models for which numbers are reported in the paper

`artifacts/human_eval_cache.pkl` - Cache of human judgments for answers given by all the models plus the initial human answers in the TellMeWhy dataset. This covers answers only for the hidden test set split and is used to calculate all the Likert score metrics. It contains ~7000 answers.

`artifacts/hidden_test_set_ontology.csv` - This file contains the hidden test set where the questions are annotated with the type of knowledge needed to answer them. The ontology is defined and discussed in the paper.

## Citation

Please use the bibtex below to cite our work.

```
@inproceedings{lal-etal-2022-using,
    title = "Using Commonsense Knowledge to Answer Why-Questions",
    author = "Lal, Yash Kumar  and
      Tandon, Niket  and
      Aggarwal, Tanvi  and
      Liu, Horace  and
      Chambers, Nathanael  and
      Mooney, Raymond  and
      Balasubramanian, Niranjan",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.79",
    pages = "1204--1219",
    abstract = "Answering questions in narratives about why events happened often requires commonsense knowledge external to the text. What aspects of this knowledge are available in large language models? What aspects can be made accessible via external commonsense resources? We study these questions in the context of answering questions in the TellMeWhy dataset using COMET as a source of relevant commonsense relations. We analyze the effects of model size (T5 and GPT3) along with methods of injecting knowledge (COMET) into these models. Results show that the largest models, as expected, yield substantial improvements over base models. Injecting external knowledge helps models of various sizes, but the amount of improvement decreases with larger model size. We also find that the format in which knowledge is provided is critical, and that smaller models benefit more from larger amounts of knowledge. Finally, we develop an ontology of knowledge types and analyze the relative coverage of the models across these categories.",
}
```


## Contributors

[Yash Kumar Lal](https://www3.cs.stonybrook.edu/~ylal/), [Niket Tandon](https://allenai.org/team/nikett), [Tanvi Aggarwal](https://www.linkedin.com/in/tanviaggarwal97), [Horace Liu](https://www.linkedin.com/in/horaceliu/) [Nate Chambers](https://www.usna.edu/Users/cs/nchamber/), [Raymond Mooney](https://www.cs.utexas.edu/~mooney/), [Niranjan Balasubramanian](https://www3.cs.stonybrook.edu/~niranjan/)
