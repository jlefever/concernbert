import logging
import json
from dataclasses import asdict
from io import IOBase, TextIOBase

import click
import pandas as pd
from concernbert import fileranking, metrics, selection, training, frontend
from concernbert.embeddings import load_caching_embedder
from sentence_transformers import SentenceTransformer


@click.group(context_settings={"show_default": True})
def cli():
    pass


@click.command()
@click.argument("INPUT", type=click.File("r"))
@click.argument("TRAIN", type=click.File("x"))
@click.argument("TEST", type=click.File("x"))
@click.argument("VAL", type=click.File("x"))
@click.option("--test-ratio", default=0.16, help="Fraction of lines that go to test")
@click.option("--val-ratio", default=0.04, help="Fraction of lines that go to val")
@click.option("--seed", help="Seed of RNG used for splitting")
def split(
    input: TextIOBase,
    train: TextIOBase,
    test: TextIOBase,
    val: TextIOBase,
    test_ratio: float,
    val_ratio: float,
    seed: int | None,
) -> None:
    """Split the lines of a text file into train, test, and val sets.

    Distributes the lines found in INPUT into newly created files TRAIN, TEST,
    and VAL according to the provided percentages. Each line is uniquely
    assigned to one of these three files. The order of the lines is preserved.
    """
    logging.debug("split called")
    lines = input.readlines()
    lines_train, lines_test, lines_val = selection.split_lines(
        lines, test_ratio, val_ratio, seed=seed
    )
    train.write("".join(lines_train))
    test.write("".join(lines_test))
    val.write("".join(lines_val))


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.File("x"))
def extract_files(input: str, output: TextIOBase):
    """Extract a CSV listing all valid files found inside the dbs.

    Reads in each path listed in INPUT and attempts to open it as a SQLite
    database. Executes several queries to get file-level data. Writes file-level
    rows to OUTPUT.
    """
    logging.debug("extract_files called")
    db_paths = selection.list_db_paths(input)
    files_df = selection.load_multi_files_df(db_paths)
    selection.insert_ldl_cols(files_df)
    files_df.to_csv(output, index=False)  # type: ignore


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.File("xb"))
@click.option("--ldl", is_flag=True, help="Only include LDL files")
@click.option("--non-ldl", is_flag=True, help="Only include non-LDL files")
def extract_entities(input: str, output: IOBase, ldl: bool, non_ldl: bool) -> None:
    """Extract a parquet table of entities.

    Given an INPUT file created by extract-files, this will create a parquet
    file at OUTPUT where each row is an entity from one of the files mentioned
    in INPUT. Only entities with at least one sibling will be output.
    """
    logging.debug("extract_entities called")
    if ldl and non_ldl:
        raise click.UsageError("Cannot use --ldl and --non-ldl together.")
    files_df = pd.read_csv(input)
    if ldl:
        files_df = files_df[files_df["is_ldl"]]
    elif non_ldl:
        files_df = files_df[~files_df["is_ldl"]]
    entities_df = selection.extract_entities_df(files_df, pbar=True)
    entities_df.to_parquet(output, index=False)  # type:ignore


@click.command()
@click.argument("CONFIG_FILE", type=click.Path(exists=True, dir_okay=False))
def train(config_file: str) -> None:
    """Runs the training procedure.

    Training arguments are specified in CONFIG_FILE. See config.ini for an
    example.
    """
    logging.debug("train called")
    training_args = training.TrainingArgs.from_ini(config_file)
    logging.info(f"Loaded training args: {training_args}")
    training.train(training_args)


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.Path(exists=False))
@click.option("--model", type=click.Path(exists=True))
@click.option("--cache", type=click.Path())
@click.option("--batch-size", default=24, type=click.INT)
def report_metrics(input: str, output: str, model: str, cache: str, batch_size):
    logging.debug("report_metrics called")
    files_df = pd.read_csv(input)
    metrics_df = metrics.calc_metrics_df(files_df, model, cache, batch_size, pbar=True)
    metrics_df.to_csv(output, index=False)


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.Path(exists=False))
@click.option("--name", help="Name of sequence")
@click.option("--seed", type=click.INT, help="Seed of RNG used when sampling pairs")
@click.option(
    "--ratio",
    type=click.FLOAT,
    default=0.01,
    help="Fraction of standard deviation of LLOC and Members column (used for tolerance)",
)
@click.option("-n", type=click.INT, default=2400, help="Number of pairs to generate")
def export_file_ranker(
    input: str, output: str, name: str, seed: int | None, ratio: float, n: int
) -> None:
    """Exports a CSV of file pairs that can be used in the fileranker web
    application."""
    logging.debug("export_file_ranker called")
    files_df = pd.read_csv(input)
    out_df = fileranking.calc_file_ranker_df(
        files_df, name=name, seed=seed, ratio=ratio, n=n
    )
    out_df.to_csv(output)


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.File("x"))
@click.option(
    "--max-pos", type=click.INT, help="Max position within a sequence to extract"
)
@click.option("--seq", help="Sequence to extract")
def extract_files_from_seq(
    input: str, output: TextIOBase, max_pos: int | None, seq: str
):
    """Extract a CSV listing all valid files found inside a sequences.csv.

    Like extract-files, but takes a sequences.csv as INPUT. The resulting OUTPUT
    csv will contain only the files mentioned in the provided sequences.csv.
    """
    logging.debug("extract_files_from_seq called")
    seq_df = pd.read_csv(input)
    seq_df = seq_df[seq_df["sequence"] == seq]
    files_df = fileranking.load_files_from_seq_df(seq_df, max_pos=max_pos)
    files_df.to_csv(output, index=False)  # type: ignore


@click.command()
@click.argument("INPUT", type=click.File("r"))
@click.option("--model", type=click.Path(exists=True))
@click.option("--cache", type=click.Path())
@click.option("--batch-size", default=24, type=click.INT)
def calculate_cd(input: TextIOBase, model: str, cache: str, batch_size: int):
    calculator = frontend.CdCalculator(model, cache, batch_size)
    text = input.read()
    result = calculator.calc_cd(text, pbar=True)
    print(json.dumps(asdict(result), indent=4))
    

cli.add_command(split)
cli.add_command(extract_files)
cli.add_command(extract_entities)
cli.add_command(train)
cli.add_command(report_metrics)
cli.add_command(export_file_ranker)
cli.add_command(extract_files_from_seq)
cli.add_command(calculate_cd)
