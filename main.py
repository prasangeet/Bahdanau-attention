from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from model_clases import attention as scratch_attention
from model_clases import attention_pytorch as torch_attention
from src.evaluator import Evaluator
from src.experiment import prepare_experiment_data
from src.logger import setup_logger
from src.trainer import Trainer


def main():
    """Train the scratch and PyTorch attention models one after the other."""
    dataset_path = Path("dataset/english_french.csv")
    logs_dir = Path("logs")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("models") / run_name
    logger = setup_logger(logs_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run data/en-fr.py first."
        )

    batch_size = 32
    emb_dim = 128
    hidden_dim = 256
    epochs = 10
    learning_rate = 1e-3
    teacher_forcing_ratio = 0.5
    train_ratio = 0.8
    validation_ratio = 0.1
    max_samples = 50000
    random_seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Starting training run.")
    logger.info(f"Using device: {device}")
    logger.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.info(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    logger.info(f"Maximum sentence pairs: {max_samples}")
    logger.info(f"Models for this run will be written to {run_dir.resolve()}")

    experiment_data = prepare_experiment_data(
        dataset_path=dataset_path,
        batch_size=batch_size,
        max_samples=max_samples,
        random_seed=random_seed,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )

    pipeline = experiment_data["pipeline"]
    dataframe = experiment_data["dataframe"]
    train_dataloader = experiment_data["train_dataloader"]
    validation_dataloader = experiment_data["validation_dataloader"]
    test_dataloader = experiment_data["test_dataloader"]

    input_dim = len(pipeline.eng_word2idx)
    output_dim = len(pipeline.fre_word2idx)
    sos_idx = pipeline.fre_word2idx["<sos>"]
    eos_idx = pipeline.fre_word2idx["<eos>"]
    pad_idx = pipeline.fre_word2idx["<pad>"]

    logger.info(f"Loaded {len(dataframe)} sentence pairs.")
    logger.info(
        f"English vocab: {input_dim} | French vocab: {output_dim} | Device: {device}"
    )
    logger.info(
        "Train samples: "
        f"{len(experiment_data['train_indices'])} | "
        f"Validation samples: {len(experiment_data['validation_indices'])} | "
        f"Test samples: {len(experiment_data['test_indices'])}"
    )
    logger.info(f"Logs will be written to {logs_dir.resolve()}")

    model_specs = [
        {
            "name": "scratch",
            "module": scratch_attention,
        },
        {
            "name": "pytorch",
            "module": torch_attention,
        },
    ]

    for model_spec in model_specs:
        model_name = model_spec["name"]
        model_module = model_spec["module"]
        model_dir = run_dir / model_name
        checkpoint_path = model_dir / "model.pth"

        logger.info(f"Starting model: {model_name}")

        attention = model_module.Attention(hidden_dim)
        encoder = model_module.Encoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
        )
        decoder = model_module.Decoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            attention=attention,
        )

        optimizer = optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=learning_rate,
        )
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

        trainer = Trainer(
            encoder=encoder,
            decoder=decoder,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
            logger=logger,
        )

        evaluator = Evaluator(
            encoder=trainer.encoder,
            decoder=trainer.decoder,
            dataloader=validation_dataloader,
            target_idx2word=pipeline.fre_idx2word,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            pad_idx=pad_idx,
            device=device,
        )

        encoder_device = next(trainer.encoder.parameters()).device
        decoder_device = next(trainer.decoder.parameters()).device
        first_train_src, first_train_trg = next(iter(train_dataloader))

        logger.info(f"[{model_name}] Encoder parameter device: {encoder_device}")
        logger.info(f"[{model_name}] Decoder parameter device: {decoder_device}")
        logger.info(
            f"[{model_name}] First train batch source device before transfer: {first_train_src.device}"
        )
        logger.info(
            f"[{model_name}] First train batch target device before transfer: {first_train_trg.device}"
        )
        logger.info(f"[{model_name}] Trainer transfer target device: {trainer.device}")
        logger.info(f"[{model_name}] Best model will be written to {checkpoint_path.resolve()}")

        checkpoint_payload = {
            "model_name": model_name,
            "dataset_path": str(dataset_path),
            "max_samples": max_samples,
            "random_seed": random_seed,
            "train_ratio": train_ratio,
            "validation_ratio": validation_ratio,
            "batch_size": batch_size,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "device_used": device,
            "eng_word2idx": pipeline.eng_word2idx,
            "fre_word2idx": pipeline.fre_word2idx,
            "eng_idx2word": pipeline.eng_idx2word,
            "fre_idx2word": pipeline.fre_idx2word,
            "train_size": len(experiment_data["train_indices"]),
            "validation_size": len(experiment_data["validation_indices"]),
            "test_size": len(experiment_data["test_indices"]),
            "test_dataloader_batch_size": len(test_dataloader),
        }

        trainer.train(
            epochs=epochs,
            evaluator=evaluator,
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload,
        )

        final_metrics = evaluator.evaluate_bleu(split_name=f"{model_name}-validation")
        logger.info(
            f"[{model_name}] Final validation accuracy: {final_metrics['accuracy']:.4f} | "
            f"Final validation BLEU: {final_metrics['bleu']:.4f}"
        )

    logger.info(
        "Training finished for both models. Run `uv run test_models.py` "
        "to compare them on the held-out test split and save attention plots."
    )


if __name__ == "__main__":
    main()
