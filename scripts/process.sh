#!/bin/sh

if [ -z "$VIRTUAL_ENV" ]; then
	echo "Please activate the virtual environment and install dependencies before running this script."
	exit 1
fi

dataset="../whyqa/dataset/test.jsonl"
converted_file="data/converted.json"
converted_dir="data/converted"

if [ ! -f "$dataset" ]; then
	echo "The file ../whyqa/dataset/test.jsonl does not exist."
	echo "Clone the WhyQA repository in the parent directory and download the dataset."
	echo "  https://github.com/oyarsa/whyqa"
	exit 1
fi

python src/convert.py artifacts/human_eval_cache.pkl "$dataset" "$converted_file"
python src/split.py "$converted_file" "$converted_dir"
