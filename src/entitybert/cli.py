import logging
from io import IOBase, TextIOBase

import click
import pandas as pd
from entitybert import fileranking, metrics, selection, training
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s][%(module)s:%(lineno)d %(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler()],
)


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
    training_args = training.TrainingArgs.from_ini(config_file)
    logging.info(f"Loaded training args: {training_args}")
    training.train(training_args)


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.Path(exists=False))
@click.option("--model", type=click.Path(exists=True))
def report_metrics(input: str, output: str, model: str):
    files_df = pd.read_csv(input)
    model_obj = SentenceTransformer(model)
    with pd.ExcelWriter(output) as writer:
        logging.info(f"Calculating metrics for {len(files_df)} files...")
        metrics_df = metrics.calc_metrics_df(files_df, model_obj, pbar=True)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)
        logging.info("Calculating correlations...")
        coef_df = metrics.calc_db_level_coefs(metrics_df)
        coef_df.to_excel(writer, sheet_name="DB-level Correlations", index=False)


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
    files_df = pd.read_csv(input)
    out_df = fileranking.calc_file_ranker_df(
        files_df, name=name, seed=seed, ratio=ratio, n=n
    )
    out_df.to_csv(output)


cli.add_command(split)
cli.add_command(extract_files)
cli.add_command(extract_entities)
cli.add_command(train)
cli.add_command(report_metrics)
cli.add_command(export_file_ranker)
