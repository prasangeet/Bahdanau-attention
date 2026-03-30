from pathlib import Path
import random

from src.dataloader import DataLoaderModule
from src.preprocessing import PreprocessingPipeline


def _select_sequences(sequences, indices):
    """Pick a subset of sequences by index while preserving their order."""
    return [sequences[index] for index in indices]


def prepare_experiment_data(
    dataset_path,
    batch_size,
    max_samples,
    random_seed,
    train_ratio,
    validation_ratio,
):
    """Load the dataset, build vocabularies, and create train/val/test splits."""
    dataset_path = Path(dataset_path)
    pipeline = PreprocessingPipeline(str(dataset_path), max_samples=max_samples)
    dataframe = pipeline.load_data()
    pipeline.build_vocab()
    eng_data, fre_data = pipeline.process_data()

    total_samples = len(eng_data)
    indices = list(range(total_samples))
    random.Random(random_seed).shuffle(indices)

    train_end = max(1, int(total_samples * train_ratio))
    validation_end = max(train_end + 1, int(total_samples * (train_ratio + validation_ratio)))

    train_indices = indices[:train_end]
    validation_indices = indices[train_end:validation_end]
    test_indices = indices[validation_end:]

    # Keep the splits usable even for small experimental subsets.
    if not validation_indices:
        validation_indices = train_indices[-1:]
        train_indices = train_indices[:-1]

    if not test_indices:
        test_indices = validation_indices[-1:]
        validation_indices = validation_indices[:-1]

    train_eng_data = _select_sequences(eng_data, train_indices)
    train_fre_data = _select_sequences(fre_data, train_indices)
    validation_eng_data = _select_sequences(eng_data, validation_indices)
    validation_fre_data = _select_sequences(fre_data, validation_indices)
    test_eng_data = _select_sequences(eng_data, test_indices)
    test_fre_data = _select_sequences(fre_data, test_indices)

    train_dataloader = DataLoaderModule(
        eng_data=train_eng_data,
        fre_data=train_fre_data,
        batch_size=batch_size,
        shuffle=True,
    ).get_loader()

    validation_dataloader = DataLoaderModule(
        eng_data=validation_eng_data,
        fre_data=validation_fre_data,
        batch_size=batch_size,
        shuffle=False,
    ).get_loader()

    test_dataloader = DataLoaderModule(
        eng_data=test_eng_data,
        fre_data=test_fre_data,
        batch_size=batch_size,
        shuffle=False,
    ).get_loader()

    return {
        "pipeline": pipeline,
        "dataframe": dataframe,
        "train_dataloader": train_dataloader,
        "validation_dataloader": validation_dataloader,
        "test_dataloader": test_dataloader,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_dataframe": dataframe.iloc[train_indices].reset_index(drop=True),
        "validation_dataframe": dataframe.iloc[validation_indices].reset_index(drop=True),
        "test_dataframe": dataframe.iloc[test_indices].reset_index(drop=True),
    }
