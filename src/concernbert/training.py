import configparser
import csv
import datetime
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from statistics import mean

import numpy as np
import pandas as pd
from concernbert.embeddings import RawEmbedder
from concernbert.sampling import MyDataset, SamplerArgs
from sentence_transformers import SentenceTransformer, losses  # type: ignore
from sentence_transformers.evaluation import SentenceEvaluator  # type: ignore
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from torch.utils.data import DataLoader

cosine_distance = losses.BatchHardTripletLossDistanceFunction.cosine_distance  # type: ignore
eucledian_distance = losses.BatchHardTripletLossDistanceFunction.eucledian_distance  # type: ignore
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class TrainingArgs:
    seed: int
    models_dir: Path
    dataset_train_path: Path
    dataset_val_path: Path
    output_model_name: str
    base_model_name: str
    use_cosine: bool
    learning_rate: float
    weight_decay: float
    epochs: int
    min_labels: int
    max_points_per_label: int
    checkpoint_steps: int
    checkpoint_limit: int
    val_files: int
    val_batch_size: int

    @staticmethod
    def from_ini(path: str | PathLike[str]) -> "TrainingArgs":
        config = configparser.ConfigParser()
        config.read(path)
        return TrainingArgs(
            seed=int(config["global"]["seed"]),
            models_dir=Path(config["io"]["models_dir"]),
            dataset_train_path=Path(config["io"]["dataset_train_path"]),
            dataset_val_path=Path(config["io"]["dataset_val_path"]),
            output_model_name=config["io"]["output_model_name"],
            base_model_name=config["training"]["base_model_name"],
            use_cosine=bool(int(config["training"]["use_cosine"])),
            learning_rate=float(config["training"]["learning_rate"]),
            weight_decay=float(config["training"]["weight_decay"]),
            epochs=int(config["training"]["epochs"]),
            min_labels=int(config["training"]["min_labels"]),
            max_points_per_label=int(config["training"]["max_points_per_label"]),
            checkpoint_steps=int(config["validation"]["checkpoint_steps"]),
            checkpoint_limit=int(config["validation"]["checkpoint_limit"]),
            val_files=int(config["validation"]["val_files"]),
            val_batch_size=int(config["validation"]["val_batch_size"]),
        )

    def sampler_args(self) -> SamplerArgs:
        return SamplerArgs(
            seed=self.seed,
            epochs=self.epochs,
            min_labels=self.min_labels,
            max_points_per_label=self.max_points_per_label,
        )


def select_rand_parent_group(df: pd.DataFrame, group_size: int) -> pd.DataFrame | None:
    """Return all rows that share a randomly chosen parent_id"""
    parent_ids = df["parent_id"].unique()
    try:
        selected_parent_ids = np.random.choice(
            parent_ids, size=group_size, replace=False
        )
    except ValueError:
        return None
    return df[df["parent_id"].isin(selected_parent_ids)]


def select_rand_parent_groups(
    df: pd.DataFrame, group_size: int, n: int, *, uni: bool = True
) -> list[pd.DataFrame]:
    parent_group_dfs = []
    df = df.copy()
    db_paths = df["db_path"].unique()
    num_parent_ids = len(df["parent_id"].unique())
    if n > num_parent_ids:
        logging.warn(
            f"Entering infinite loop, n ({n}) is greater than {num_parent_ids}"
        )
    while True:
        df_input = df[df["db_path"] == random.choice(db_paths)] if uni else df
        parent_group_df = select_rand_parent_group(df_input, group_size=group_size)
        if parent_group_df is None:
            continue
        parent_group_dfs.append(parent_group_df)
        df.drop(parent_group_df.index, inplace=True)
        if len(parent_group_dfs) >= n:
            break
    return parent_group_dfs


class MyEvaluator(SentenceEvaluator):
    def __init__(
        self,
        filename: str,
        df: pd.DataFrame,
        batch_size: int,
        group_sizes: list[int],
        n: int,
        seed: int,
    ):
        self._filename = filename
        self._df = df
        self._group_sizes = group_sizes
        self._seed = seed
        self._batch_size = batch_size
        self._csv_headers = ["timestamp", "epoch", "steps"] + [
            f"NMI-{s}" for s in group_sizes
        ]

        random.seed(seed)
        np.random.seed(seed)
        self._groups: dict[int, list[pd.DataFrame]] = defaultdict(list)
        for group_size in group_sizes:
            groups = select_rand_parent_groups(self._df, group_size, n)
            self._groups[group_size] = groups

    def summary(self) -> str:
        lines: list[str] = []
        for group_size in self._group_sizes:
            n = len(self._groups[group_size])
            lines.append(f"Evaluator has {n} groups with {group_size} classes merged.")
        return "\n".join(lines)

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
    ) -> float:
        logging.info(f"Evaluating in epoch {epoch} after {steps} steps...")
        embedder = RawEmbedder(model, self._batch_size)

        # Calculate embeddings
        texts: list[str] = []
        for group_dfs in self._groups.values():
            for group_df in group_dfs:
                texts.extend(group_df["content"])
        embeddings = embedder.embed(texts, pbar=True)

        # Calculate scores
        scores: list[float] = []
        for group_size in self._group_sizes:
            nmis: list[float] = []
            for group_df in self._groups[group_size]:
                texts: list[str] = list(group_df["content"])  # type: ignore
                group_embeddings = [embeddings[t] for t in texts]
                try:
                    kmeans = KMeans(n_clusters=group_size, random_state=self._seed)
                    pred_labels: list[int] = list(kmeans.fit(group_embeddings).labels_)  #  type: ignore
                    true_labels: list[str] = list(group_df["parent_id"])  # type: ignore
                    nmi = normalized_mutual_info_score(pred_labels, true_labels)
                except Exception as e:
                    logging.warn(f"Exception encountered during eval: {e}")
                    continue
                nmis.append(float(nmi))
            score = mean(nmis) if len(nmis) > 0 else 0.0
            logging.info(f"NMI-{group_size}: {score}")
            scores.append(score)

        if output_path is None:
            return mean(scores) if len(scores) > 0 else 0.0

        # Write to CSV
        csv_path = os.path.join(output_path, self._filename)
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().isoformat()
        if not os.path.isfile(csv_path):
            with open(csv_path, newline="", mode="w") as f:
                writer = csv.writer(f)
                writer.writerow(self._csv_headers)
                writer.writerow([timestamp, epoch, steps] + scores)
        else:
            with open(csv_path, newline="", mode="a") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, epoch, steps] + scores)
        return mean(scores) if len(scores) > 0 else 0.0


def train(args: TrainingArgs):
    timestamp = datetime.datetime.now().isoformat()
    output_path = Path(args.models_dir, args.output_model_name)
    checkpoint_path = Path(args.models_dir, f"{args.output_model_name}-checkpoints")

    if output_path.exists():
        raise RuntimeError(f"{output_path} already exists")

    logging.info("Loading train and val dataset...")
    df_train = pd.read_parquet(args.dataset_train_path)
    df_val = pd.read_parquet(args.dataset_val_path)

    logging.info("Setting up evaluator...")
    evaluator = MyEvaluator(
        f"{timestamp}.csv", df_val, args.val_batch_size, [2], args.val_files, args.seed
    )
    logging.info(evaluator.summary())

    logging.info("Setting up dataloader and sampler...")
    dataset = MyDataset(df_train)
    sampler = dataset.sampler(args.sampler_args())
    logging.info(sampler.summary())
    dataloader = DataLoader(dataset, batch_sampler=sampler)

    logging.info("Loading fresh model...")
    model = SentenceTransformer(args.base_model_name)
    dist = cosine_distance if args.use_cosine else eucledian_distance  # type: ignore
    train_loss = losses.BatchHardSoftMarginTripletLoss(model, dist)

    # Disable this because it seems to cause NaNs during training
    # train_loss = losses.Matryoshka2dLoss(model, train_loss, [768, 512, 256, 128, 64])

    logging.info(f"device: {model.device}")

    logging.info("Checking performance before fine-tuning...")
    eval_dir = Path(args.models_dir, args.output_model_name, "eval")
    evaluator(model, str(eval_dir), epoch=0, steps=0)

    logging.info("Start training...")
    model.fit(  # type: ignore
        train_objectives=[(dataloader, train_loss)],
        evaluator=evaluator,
        epochs=sampler.n_epochs(),
        steps_per_epoch=sampler.n_batches_per_epoch(),
        evaluation_steps=args.checkpoint_steps,
        scheduler="WarmupLinear",
        warmup_steps=int(sampler.n_batches() * 0.10),
        optimizer_params={"lr": args.learning_rate},
        weight_decay=args.weight_decay,
        use_amp=True,
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        checkpoint_save_steps=args.checkpoint_steps,
        checkpoint_save_total_limit=args.checkpoint_limit,
    )
