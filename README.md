# EntityBERT

Source code embeddings from finetuned BERT-based models.

## Dependencies

This project is managed with [Rye](https://github.com/astral-sh/rye).

## Usage

```bash
rye sync # only needs to be run once
source .venv/bin/activate
python -m entitybert --help
```

## Examples

Below is an example of building a dataset.

```bash
python -m entitybert split --seed 42 _data/dbs.txt _data/dbs_train.txt _data/dbs_test.txt _data/dbs_val.txt

python -m entitybert extract-files _data/dbs_val.txt _data/files_val.csv
python -m entitybert extract-files _data/dbs_test.txt _data/files_test.csv
python -m entitybert extract-files _data/dbs_train.txt _data/files_train.csv

python -m entitybert extract-entities _data/files_test.csv _data/entities_test_all.parquet
python -m entitybert extract-entities --ldl _data/files_test.csv _data/entities_test_ldl.parquet
python -m entitybert extract-entities --non-ldl _data/files_test.csv _data/entities_test_nonldl.parquet

python -m entitybert extract-entities _data/files_train.csv _data/entities_train_all.parquet
python -m entitybert extract-entities --ldl _data/files_train.csv _data/entities_train_ldl.parquet
python -m entitybert extract-entities --non-ldl _data/files_train.csv _data/entities_train_nonldl.parquet

python -m entitybert extract-entities _data/files_val.csv _data/entities_val_all.parquet
python -m entitybert extract-entities --ldl _data/files_val.csv _data/entities_val_ldl.parquet
python -m entitybert extract-entities --non-ldl _data/files_val.csv _data/entities_val_nonldl.parquet
```

Below can be used for generating input to the fileranker web application

```bash
python -m entitybert export-file-ranker --name testset-ldl --seed 42 _data/files_test.csv testset-ldl.csv
```

Below can be used for generating a metics report

```bash
python -m entitybert report-metrics --model _models/my_model/ _data/files_dummy.csv _data/metrics_dummy.xlsx
```
