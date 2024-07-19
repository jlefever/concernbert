# EntityBERT

Source code embeddings from finetuned BERT-based models.

## Dependencies

This project is managed with [Rye](https://github.com/astral-sh/rye).

## Usage

```bash
rye sync # only needs to be run once
.venv/bin/activate
python -m entitybert --help
```

## Collecting Data

Below is an example of building a dataset.

```bash
python -m entitybert split --seed 42 _data/dbs.txt _data/dbs_train.txt _data/dbs_test.txt _data/dbs_val.txt

python -m entitybert extract-files _data/dbs_val.txt _data/files_val.csv
python -m entitybert extract-files _data/dbs_test.txt _data/files_test.csv
python -m entitybert extract-files _data/dbs_train.txt _data/files_train.csv

python -m entitybert filter-files _data/files_val.csv _data/files_val_filtered.csv
python -m entitybert filter-files _data/files_test.csv _data/files_test_filtered.csv
python -m entitybert filter-files _data/files_train.csv _data/files_train_filtered.csv

python -m entitybert extract-entities _data/files_val_filtered.csv _data/entities_val.parquet
python -m entitybert extract-entities _data/files_test_filtered.csv _data/entities_test.parquet
python -m entitybert extract-entities _data/files_train_filtered.csv _data/entities_train.parquet
```
