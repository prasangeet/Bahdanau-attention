import json
from pathlib import Path
import numpy as np

import torch

from src.attention_plotter import save_attention_plot
from src.evaluator import Evaluator
from src.experiment import prepare_experiment_data
from src.logger import setup_logger


def _find_latest_run(models_dir):
    """Pick the most recent training run directory."""
    run_directories = [path for path in Path(models_dir).iterdir() if path.is_dir()]

    if not run_directories:
        raise FileNotFoundError("No model run directories were found in models/.")

    return sorted(run_directories)[-1]


def _load_checkpoint(checkpoint_path, device):
    """Load a full-model checkpoint onto the requested device."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint["encoder"] = checkpoint["encoder"].to(device)
    checkpoint["decoder"] = checkpoint["decoder"].to(device)
    return checkpoint


def main():
    """Evaluate both saved models on the test split and save attention plots."""
    logs_dir = Path("logs")
    logger = setup_logger(logs_dir, log_name="test_models.log")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = _find_latest_run("models")

    logger.info(f"Comparing saved models from run: {run_dir.resolve()}")
    logger.info(f"Evaluation device: {device}")

    scratch_checkpoint = _load_checkpoint(run_dir / "scratch" / "model.pth", device)
    pytorch_checkpoint = _load_checkpoint(run_dir / "pytorch" / "model.pth", device)
    reference_checkpoint = scratch_checkpoint

    experiment_data = prepare_experiment_data(
        dataset_path=reference_checkpoint["dataset_path"],
        batch_size=reference_checkpoint["batch_size"],
        max_samples=reference_checkpoint["max_samples"],
        random_seed=reference_checkpoint["random_seed"],
        train_ratio=reference_checkpoint["train_ratio"],
        validation_ratio=reference_checkpoint["validation_ratio"],
    )

    test_dataloader = experiment_data["test_dataloader"]
    test_dataframe = experiment_data["test_dataframe"]
    pipeline = experiment_data["pipeline"]
    plot_dir = run_dir / "attention_plots"
    results = {}

    checkpoints = {
        "scratch": scratch_checkpoint,
        "pytorch": pytorch_checkpoint,
    }

    for model_name, checkpoint in checkpoints.items():
        evaluator = Evaluator(
            encoder=checkpoint["encoder"],
            decoder=checkpoint["decoder"],
            dataloader=test_dataloader,
            target_idx2word=checkpoint["fre_idx2word"],
            sos_idx=checkpoint["fre_word2idx"]["<sos>"],
            eos_idx=checkpoint["fre_word2idx"]["<eos>"],
            pad_idx=checkpoint["fre_word2idx"]["<pad>"],
            device=device,
        )

        metrics = evaluator.evaluate_bleu(split_name=f"{model_name}-test")
        results[model_name] = metrics
        logger.info(
            f"[{model_name}] Test accuracy: {metrics['accuracy']:.4f} | "
            f"Test BLEU: {metrics['bleu']:.4f}"
        )

        # Select 3 random sentences from the test dataset
        num_examples = min(3, len(test_dataframe))
        random_indices = np.random.choice(len(test_dataframe), size=num_examples, replace=False)
        
        for example_num, test_index in enumerate(random_indices, 1):
            row = test_dataframe.iloc[test_index]
            src_tokens = pipeline.numricalize(row["English"], checkpoint["eng_word2idx"])
            predicted_words, attention_matrix = evaluator.decode_single_example(src_tokens)

            source_words = row["English"].split()
            output_path = plot_dir / model_name / f"example_{example_num}.png"
            save_attention_plot(
                source_tokens=source_words,
                target_tokens=predicted_words,
                attention_matrix=attention_matrix,
                output_path=output_path,
                title=f"{model_name} attention example {example_num}",
            )

            logger.info(
                f"[{model_name}] Saved attention plot for example {example_num} to {output_path.resolve()}"
            )

    summary_path = run_dir / "test_results.json"
    summary_path.write_text(json.dumps(results, indent=2))
    logger.info(f"Wrote test comparison summary to {summary_path.resolve()}")


if __name__ == "__main__":
    main()
