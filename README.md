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
find .data/dbs/**/*.db > .data/dbs.txt
python -m entitybert split .data/dbs.txt .data/dbs_train.txt .data/dbs_test.txt .data/dbs_val.txt

python -m entitybert extract-files .data/dbs_val.txt .data/files_val.csv
python -m entitybert extract-files .data/dbs_test.txt .data/files_test.csv
python -m entitybert extract-files .data/dbs_train.txt .data/files_train.csv

python -m entitybert filter-files .data/files_val.csv .data/files_val_filtered.csv
python -m entitybert filter-files .data/files_test.csv .data/files_test_filtered.csv
python -m entitybert filter-files .data/files_train.csv .data/files_train_filtered.csv

python -m entitybert extract-entities .data/files_val_filtered.csv .data/entities_val.parquet
python -m entitybert extract-entities .data/files_test_filtered.csv .data/entities_test.parquet
python -m entitybert extract-entities .data/files_train_filtered.csv .data/entities_train.parquet
```
