from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import faiss
from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Union
import pickle
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from dataclasses import dataclass


@dataclass
class ExperimentData:
    content: dict  # Reference, normal test, anomalous test.
    version_id: str  # A hash or timestamp


class DataProvider(ABC):
    """Abstract base class for providing experiment data. Implementations can be static or dynamic.
    get_version_id is used to determine if the data has changed and if the experiment needs to be rebuilt.
    get_data returns the current data for the experiment. refresh_data can be used to update the data in dynamic providers.
    """

    @abstractmethod
    def get_version_id(self) -> str:
        """Returns a unique identifier for the current state of data."""
        pass

    @abstractmethod
    def get_data(self) -> ExperimentData:
        pass

    @abstractmethod
    def refresh_data(self, new_data: dict):
        """Updates the data in the provider."""
        pass


class StaticDataProvider(DataProvider):
    def __init__(self, data: dict):
        self.data = data
        # Current version_id implementation assumes data is a dict of lists.
        self.version_id = str(sum(hash(tuple(val)) for val in data.values()))

    def get_version_id(self) -> str:
        return self.version_id

    def get_data(self) -> ExperimentData:
        return ExperimentData(content=self.data, version_id=self.version_id)

    def refresh_data(self):
        pass


class RandomDataProvider(DataProvider):
    def __init__(self, df, sizes, ref_df_len=None, seed=42):
        self.sizes = sizes  # of entries in each category (reference, normal test, anomalous test)
        self.df = df  # Polars DataFrame. Cols 'entry' and 'is_normal' are required.
        self.seed = seed
        self._counter = 0
        # Optional: if provided, the provided `df` is a concatenation of the
        # reference dataframe and the test dataframe (in that order). When
        # `ref_df_len` is set, reference entries will be drawn from the first
        # `ref_df_len` rows and test entries will be drawn from the remaining rows.
        self.ref_df_len = ref_df_len

        if self.ref_df_len is not None:
            total_rows = self.df.height
            if self.ref_df_len < 0 or self.ref_df_len > total_rows:
                raise ValueError("ref_df_len must be between 0 and number of rows in df")
            self._ref_normal_pool = self.df.head(self.ref_df_len)
            self._test_normal_pool = self.df.tail(total_rows - self.ref_df_len).filter(pl.col("is_normal") == 1)
            self._test_anom_pool = self.df.tail(total_rows - self.ref_df_len).filter(pl.col("is_normal") == 0)
        else:
            self._ref_normal_pool = self.df.filter(pl.col("is_normal") == 1)
            self._test_normal_pool = self.df.filter(pl.col("is_normal") == 1)
            self._test_anom_pool = self.df.filter(pl.col("is_normal") == 0)

        self.refresh_data()

    def get_version_id(self) -> str:
        return self.version_id

    def get_data(self) -> ExperimentData:
        return ExperimentData(content=self.data, version_id=self.version_id)

    def refresh_data(self):
        # Generate random data from self.df, with the subset sizes specified in self.sizes.
        # Use self.seed and self._counter to ensure different random samples across calls.
        self._counter += 1
        seed = self.seed + self._counter

        # Helper that decides whether to sample with replacement if pool too small
        def _sample(pool, desired):
            pool_size = pool.height
            if pool_size == 0:
                return pl.DataFrame({"entry": []})
            with_replacement = desired > pool_size
            return pool.sample(n=desired, seed=seed, with_replacement=with_replacement)

        ref_sample = _sample(self._ref_normal_pool, self.sizes["reference"])
        normal_sample = _sample(self._test_normal_pool, self.sizes["test_normal"])
        anomalous_sample = _sample(self._test_anom_pool, self.sizes["test_anomalous"])

        random_data = {
            "reference": ref_sample["entry"].to_numpy(),
            "test_normal": normal_sample["entry"].to_numpy(),
            "test_anomalous": anomalous_sample["entry"].to_numpy(),
        }
        self.data = random_data
        self.version_id = str(time.time())


class Experiment:
    def __init__(
        self,
        embedding_method: Embedder,
        nlist: int,
        nprobe: int,
        data_provider: DataProvider,
        k: int,
        embeddings_dir: str,
        index_type: str = "ivf",
        device: str = "auto",
    ):
        self.embedding_method = embedding_method
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_type = index_type
        self.data_provider = data_provider
        self.data = None
        self.embeddings_dir = embeddings_dir
        self.embedded_reference = None
        self.embedded_test_normal = None
        self.embedded_test_anomalous = None
        self.k = k
        self.is_built = False
        self.build_id = None
        self.device = device

    def build(self, measure_time=False, refresh_data=True):
        """Builds the experiment by embedding the reference and test entries."""
        # If build_id matches the current data version,
        # the embeddings are already present.
        data_version_id = self.data_provider.get_version_id()
        embeddings_id = f"embeddings_{self.embedding_method.name}_{data_version_id}"
        embeddings_path = os.path.join(self.embeddings_dir, embeddings_id + ".pkl")
        path_exists = os.path.exists(embeddings_path)
        if self.build_id == data_version_id and not measure_time:
            print(
                "Experiment is already built with the current data version. Skipping build."
            )
            return
        self.data = self.data_provider.get_data().content
        self.reference_entries = self.data["reference"]
        self.test_normal_entries = self.data["test_normal"]
        self.test_anomalous_entries = self.data["test_anomalous"]
        if path_exists and not measure_time:
            with open(embeddings_path, "rb") as f:
                self.data = pickle.load(f)
                self.embedded_reference = self.data["embedded_reference"]
                self.embedded_test_normal = self.data["embedded_test_normal"]
                self.embedded_test_anomalous = self.data["embedded_test_anomalous"]
        else:
            self.embedded_reference = embed_entries(
                [self.reference_entries], self.embedding_method, True
            )
            start_time = time.time()
            self.embedded_test_normal = embed_entries(
                [self.test_normal_entries], self.embedding_method, False
            )
            end_time = time.time()
            if measure_time:
                self.embed_normal_time = end_time - start_time
            start_time = time.time()
            self.embedded_test_anomalous = embed_entries(
                [self.test_anomalous_entries],
                self.embedding_method,
                False,
            )
            end_time = time.time()
            if measure_time:
                self.embed_anomalous_time = end_time - start_time
            # Save the embeddings for future use unless they exist.
            # (In which case they were re-embedded for timing purposes.)
            # Create directory if not present.
            if not path_exists:
                os.makedirs(self.embeddings_dir, exist_ok=True)
                with open(embeddings_path, "wb") as f:
                    pickle.dump(
                        {
                            "embedded_reference": self.embedded_reference,
                            "embedded_test_normal": self.embedded_test_normal,
                            "embedded_test_anomalous": self.embedded_test_anomalous,
                        },
                        f,
                    )
        self.is_built = True
        self.build_id = data_version_id


def create_extended_entries(entries,
                            window_size,
                            concat_symbol="|",
                            ):
    """Creates extended entries by concatenating at most window_size consecutive log
    entries. The entries are joined with the specified concat_symbol.
    Extended entries have a stride of one; i.e. there is one extended entry for each
    original entry. At the tail end, extended entries terminate at the end of the
    entry list even if the window is not full.
    NOTE: This function does not add start/end tokens as this would change the number
    of entries."""
    extended_entries = []
    for i in range(len(entries)):
        extended_entry = concat_symbol.join(entries[i : i + window_size])
        extended_entries.append(extended_entry)
    return extended_entries


def embed_entries(
    extended_entry_lists,
    embedder,
    is_reference=False
):
    if is_reference:
        c = Counter()
        for extended_entry_list in extended_entry_lists:
            c.update(extended_entry_list)
        entry_df = pl.DataFrame(
            {"extended_entry": list(c.keys()), "count": list(c.values())}
        )
    else:
        entry_df = pl.DataFrame(
            {
                "entry": extended_entry_lists[0],
                "extended_entry": extended_entry_lists[0],
            }
        )
    if hasattr(embedder, "encode"):
        embeddings = embedder.encode(entry_df["extended_entry"].to_list())
    else:
        embeddings = embedder.embed(entry_df["extended_entry"].to_list())
    entry_df = entry_df.with_columns(pl.Series("embedding", embeddings))
    return entry_df


def create_faiss_index(
    ref_embeddings_df,
    index_type: str = "ivf",
    nlist: int = 100,
    gpu_res=None,
):
    """Create a FAISS index from reference embeddings.

    Args:
        ref_embeddings_df: polars DataFrame with an "embedding" column (and "count" for IVF).
        index_type: Either "flat" or "ivf". If "flat", `nlist` is ignored.
        nlist: Number of IVF lists (only used when ``index_type`` is not "flat").
        gpu_res: Optional FAISS GPU resources.
    """
    if not isinstance(ref_embeddings_df, pl.DataFrame):
        raise ValueError("ref_embeddings_df must be a polars DataFrame")

    embeddings = ref_embeddings_df["embedding"].to_numpy()
    d = embeddings.shape[1]

    # Validate index_type and nlist
    if index_type not in ("flat", "ivf"):
        raise ValueError("index_type must be 'flat' or 'ivf'")
    if index_type != "flat":
        if nlist is None or nlist <= 0:
            raise ValueError("nlist must be a positive integer for IVF indexes")

    if index_type == "flat":
        index = faiss.IndexFlatL2(d)
        if gpu_res is not None:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        index.add(embeddings)
        return index

    # For IVF indices, use the provided nlist.
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    start_time = time.time()
    print(f"Training the Faiss index on the reference embeddings...")
    index.train(embeddings)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")

    if gpu_res is not None:
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    start_time = time.time()
    print(f"Adding the reference embeddings to the Faiss index...")
    index.add(embeddings)
    print(f"Adding completed in {time.time() - start_time:.2f} seconds.")
    return index


def calculate_distances(
    inference_entries_df, index, ref_embeddings_df, k, nprobe=10, gpu_res=None, verbose=False
):
    # Assume there is only one log type
    inference_entries_unique = inference_entries_df.unique(subset=["extended_entry"])
    # If GPU resources are provided, try to move the index to GPU for faster search.
    if gpu_res is not None:
        if verbose:
            print("Attempting to move FAISS index to GPU for faster search...")
        try:
            index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
        except Exception:
            # Some index types or FAISS builds may not support conversion; ignore and continue on CPU.
            pass

    # Only set nprobe for indexes that support it (e.g., IVF). Flat indices do not use nprobe.
    if hasattr(index, "nprobe"):
        try:
            index.nprobe = nprobe
        except Exception:
            # Some wrapped GPU/index types may not allow direct assignment; ignore if so.
            pass

    if verbose:
        print(f"Calculating distances for {len(inference_entries_unique)} unique extended entries...")
        start_time = time.time()
    D, I = index.search(inference_entries_unique["embedding"].to_numpy(), k)
    if verbose:
        print(f"Distance calculation completed in {time.time() - start_time:.2f} seconds. Computing mean distances...")
        start_time = time.time()
    mean_dists = get_mean_distances(D, I, ref_embeddings_df, k)
    if verbose:
        print(f"Mean distance calculation completed in {time.time() - start_time:.2f} seconds. Merging results...")
    inference_entries_unique = inference_entries_unique.with_columns(
        pl.Series("dist", mean_dists)
    )
    inference_entries_unique = inference_entries_unique[["extended_entry", "dist"]]
    ie_merged = inference_entries_df.join(
        inference_entries_unique, on="extended_entry", how="inner"
    )
    return ie_merged


def get_mean_distances(D, I, df_ref_embeddings, k):
    # Extract counts as NumPy array (critical for speed)
    counts = df_ref_embeddings["count"].to_numpy()

    # Shape: (n_queries, k)
    counts_per_neighbor = counts[I]

    # Cumulative counts per row
    cum_counts = np.cumsum(counts_per_neighbor, axis=1)

    # We want how many from each neighbor we actually take
    # Clip cumulative counts at k
    cum_counts_clipped = np.minimum(cum_counts, k)

    # Prepend zeros for diff
    cum_counts_shifted = np.hstack([
        np.zeros((cum_counts.shape[0], 1), dtype=cum_counts.dtype),
        cum_counts_clipped[:, :-1]
    ])

    # Number taken from each neighbor
    taken = cum_counts_clipped - cum_counts_shifted

    # Weighted sum of distances
    weighted_sum = np.sum(taken * D, axis=1)

    return weighted_sum / k


class Embedder(ABC):
    """Base class for converting strings into numerical vectors."""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Standardized method to return an embedding (or batch of embeddings)."""
        pass

    def __call__(self, text: Union[str, List[str]]) -> np.ndarray:
        """Allows the instance to be used like a function."""
        return self.embed(text)


class FastTextEmbedder(Embedder):
    """Embedder that uses a fastText model."""

    def __init__(self, model):
        super().__init__(f"fasttext_{model.get_dimension()}d")
        self.model = model

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Returns the embedding for the given text."""
        if isinstance(text, str):
            return self.model.get_sentence_vector(text)
        elif isinstance(text, list):
            return np.array([self.model.get_sentence_vector(t) for t in text])
        elif isinstance(text, np.ndarray):
            return np.array([self.model.get_sentence_vector(t) for t in text.tolist()])
        else:
            raise ValueError("Input must be a string or a list of strings.")


class SentenceTransformerEmbedder(Embedder):
    """Embedder that uses a SentenceTransformer model."""

    def __init__(self, model_or_name: Union[str, object]):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Please install the sentence-transformers library to use this embedder."
            )

        if isinstance(model_or_name, str):
            model_name = model_or_name
            self.model = SentenceTransformer(model_name)
        elif isinstance(model_or_name, SentenceTransformer):
            self.model = model_or_name
            model_name = getattr(self.model, "name_or_path", self.model.__class__.__name__)
        else:
            raise TypeError(
                "model_or_name must be either a model name (str) or a SentenceTransformer instance."
            )

        super().__init__(model_name)

    def embed(self, text: Union[str, List[str]]) -> np.ndarray:
        """Returns the embedding for the given text."""
        return self.model.encode(text)

    
def get_best_f1(y_true, y_scores):
    """Helper function to get the best F1 score and corresponding threshold."""

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_index = np.argmax(f1_scores)
    return f1_scores[best_index], thresholds[best_index]


def run_benchmark(
    experiment: Experiment,
    measure_embedding_time=False,
    reps=1,
    test_sizes=None,
    refresh_data=False,
):
    """Runs the benchmark for the given experiment. If measure_embedding_time is True, it will also measure the time taken to embed the test entries."""

    print(f"Running benchmark for embedding method: {experiment.embedding_method.name}")
    # print(f"Number of reference entries: {len(experiment.reference_entries)}")
    # print(f"Number of normal test entries: {len(experiment.test_normal_entries)}")
    # print(f"Number of anomalous test entries: {len(experiment.test_anomalous_entries)}")
    print(f"nlist: {experiment.nlist}, nprobe: {experiment.nprobe}, k: {experiment.k}")

    if test_sizes is None:
        test_sizes = [1, 10, 100, 1000]
    auroc_scores = []
    auprc_scores = []
    f1_scores = []
    dist_times_list = []
    embedding_times_normal = []
    embedding_times_anomalous = []

    for i in range(reps):
        print(f"Iteration {i+1}/{reps}...")
        results = _run_benchmark(
            experiment,
            measure_embedding_time,
            test_sizes=test_sizes,
            refresh_data=refresh_data,
        )
        auroc_scores.append(results["auroc"])
        auprc_scores.append(results["auprc"])
        f1_scores.append(results["f1"])
        dist_times_list.append(results["dist_times"])
        if measure_embedding_time:
            embedding_times_normal.append(results["embedding_time_normal"])
            embedding_times_anomalous.append(results["embedding_time_anomalous"])

    print(f"Average AUROC over {reps} iterations: {np.mean(auroc_scores):.4f}")
    print(f"Average AUPRC over {reps} iterations: {np.mean(auprc_scores):.4f}")
    print(f"Average F1 over {reps} iterations: {np.mean(f1_scores):.4f}")
    dist_times = np.array(dist_times_list)
    mean_dist_times = np.mean(dist_times, axis=0)
    print(
        f"Mean distance calculation times for test sizes {test_sizes}: {mean_dist_times}"
    )

    return {
        "auroc_scores": auroc_scores,
        "auprc_scores": auprc_scores,
        "dist_times": dist_times,
        "f1_scores": f1_scores,
        "embedding_times_normal": embedding_times_normal,
        "embedding_times_anomalous": embedding_times_anomalous,
    }


def _run_benchmark(
    experiment: Experiment,
    measure_embedding_time=False,
    test_sizes=[1, 10, 100, 1000],
    refresh_data=False,
):
    """Runs one iteration of the benchmark for the given experiment."""

    if refresh_data:
        experiment.data_provider.refresh_data()

    # Build the experiment (embed the reference and test entries).
    experiment.build(measure_time=measure_embedding_time, refresh_data=refresh_data)

    # Prepare FAISS GPU resources if requested/available and experiment wants GPU.
    gpu_res = None
    device = getattr(experiment, "device", "auto")
    if device in ("auto", "cuda", "gpu"):
        try:
            if hasattr(faiss, "StandardGpuResources"):
                gpu_res = faiss.StandardGpuResources()
        except Exception:
            gpu_res = None

    # Train the index on the reference embeddings (optionally using gpu_res).
    index = create_faiss_index(
        experiment.embedded_reference,
        index_type=getattr(experiment, "index_type", "ivf"),
        nlist=experiment.nlist,
        gpu_res=gpu_res,
    )

    # Calculate distances for normal and anomalous test entries.
    normal_scores = calculate_distances(
        experiment.embedded_test_normal,
        index,
        experiment.embedded_reference,
        experiment.k,
        nprobe=experiment.nprobe,
        gpu_res=gpu_res,
    )
    experiment.normal_scores = normal_scores
    anomalous_scores = calculate_distances(
        experiment.embedded_test_anomalous,
        index,
        experiment.embedded_reference,
        experiment.k,
        nprobe=experiment.nprobe,
        gpu_res=gpu_res,
    )
    experiment.anomalous_scores = anomalous_scores

    # Calculate AUROC.
    y_true = [1] * len(anomalous_scores) + [0] * len(normal_scores)
    y_scores = list(anomalous_scores["dist"]) + list(normal_scores["dist"])
    auroc = roc_auc_score(y_true, y_scores)

    # Calculate AUPRC.
    auprc = average_precision_score(y_true, y_scores)

    # Test distance calculation speed.
    test_sizes = [1] + test_sizes
    dist_times = []
    for size in test_sizes:
        test_subset = experiment.embedded_test_normal.head(size)
        start_time = time.time()
        _ = calculate_distances(
            test_subset,
            index,
            experiment.embedded_reference,
            experiment.k,
            nprobe=experiment.nprobe,
            gpu_res=gpu_res,
        )
        end_time = time.time()
        dist_times.append(end_time - start_time)
    # Drop the first time (warmup)
    dist_times = dist_times[1:]

    # Calculate the best F1 score.
    f1, _ = get_best_f1(y_true, y_scores)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "dist_times": dist_times,
        "f1": f1,
        "embedding_time_normal": getattr(experiment, "embed_normal_time", None),
        "embedding_time_anomalous": getattr(experiment, "embed_anomalous_time", None),
    }
