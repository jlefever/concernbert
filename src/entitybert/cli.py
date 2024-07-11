import logging
from io import IOBase, TextIOBase

import click
import pandas as pd
from entitybert import cohesion, selection, training

logger = logging.getLogger(__name__)


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
@click.option("--seed", default=42, help="Random seed used for splitting")
def split(
    input: TextIOBase,
    train: TextIOBase,
    test: TextIOBase,
    val: TextIOBase,
    test_ratio: float,
    val_ratio: float,
    seed: int,
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
@click.argument("INPUT", type=click.File("r"))
@click.argument("OUTPUT", type=click.File("x"))
def extract_files(input: TextIOBase, output: TextIOBase):
    """Extract a CSV listing all files found inside the dbs.

    Reads in each path listed in INPUT and attempts to open it as a SQLite
    database. Executes several queries to get file-level data. Writes file-level
    rows to OUTPUT.
    """
    db_paths = [line.rstrip() for line in input.readlines()]
    df = selection.load_many_files_df(db_paths, pbar=True)
    df.to_csv(output, index=False)  # type: ignore


@click.command()
@click.argument("INPUT", type=click.File("r"))
@click.argument("OUTPUT", type=click.File("x"))
@click.option("--global-quantile", default=0.80, help="The global quantile")
@click.option("--local-quantile", default=0.80, help="The local quantile")
@click.option("--keep-invalid-names", is_flag=True)
def filter_files(
    input: TextIOBase,
    output: TextIOBase,
    global_quantile: float,
    local_quantile: float,
    keep_invalid_names: bool,
) -> None:
    """Filters a CSV created with extract-files.

    Each remaining row will have a "CC (in)", "CC (out)", "CC (cross)", and "CC
    (no dep)" below both the global and local threshold for that column. The
    global threshold for a column is calculated across all rows. The local
    threshold is calculated across only the rows with the same "db_path".
    Additionally, any row with an invalid name is removed unless
    --keep-invalid-names is specified.
    """
    df = pd.read_csv(input)  # type: ignore
    df = selection.filter_files_df(
        df, global_quantile, local_quantile, keep_invalid_names
    )
    df.to_csv(output, index=False)  # type: ignore


@click.command()
@click.argument("INPUT", type=click.File("r"))
@click.argument("OUTPUT", type=click.File("xb"))
def extract_entities(input: TextIOBase, output: IOBase) -> None:
    """Extract a parquet table of entities.

    Given an INPUT file created by extract-files, this will create a parquet
    file at OUTPUT where each row is an entity from one of the files mentioned
    in INPUT. Only entities with at least one sibling will be output.
    """
    files_df = pd.read_csv(input)  # type: ignore
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
    logger.info(f"Loaded training args: {training_args}")
    training.train(training_args)


@click.command()
@click.option("--model", type=click.Path(exists=True))
@click.argument("INPUT", type=click.File("r"))
@click.argument("OUTPUT", type=click.Path(exists=False))
def report_cohesion(model: str, input: TextIOBase, output: str):
    db_paths = [line.rstrip() for line in input.readlines()]
    cohesion_df = cohesion.calc_all_cohesion_df(db_paths, model, pbar=True)
    db_level_coef_df = cohesion.calc_db_level_coefs(cohesion_df)
    with pd.ExcelWriter(output) as writer:
        cohesion_df.to_excel(writer, sheet_name="Scores", index=False)
        db_level_coef_df.to_excel(writer, sheet_name="Correlation", index=False)


@click.command()
@click.argument("INPUT", type=click.Path(exists=True))
@click.argument("OUTPUT", type=click.Path(exists=False))
def export_simple_valid_classes(input: str, output: str) -> None:
    """Exports a CSV of simple valid classes.

    Given an INPUT database file, write a CSV to OUTPUT with two columns:
    filename and content. Only includes files that are simple classes (classes
    that do not contain nested types) and have a valid name (does not appear to
    be a test.) Additionally, all classes contain at least two entities.
    """
    selection.calc_simple_valid_classes_df(input).to_csv(output)


cli.add_command(split)
cli.add_command(extract_files)
cli.add_command(filter_files)
cli.add_command(extract_entities)
cli.add_command(train)
cli.add_command(report_cohesion)
cli.add_command(export_simple_valid_classes)
