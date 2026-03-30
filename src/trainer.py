import torch
import torch.nn as nn
import random
from pathlib import Path
from tqdm import tqdm


class Trainer:
    """Manage the training loop for the encoder-decoder translation model."""

    def __init__(
        self,
        encoder,
        decoder,
        dataloader,
        optimizer,
        criterion,
        device="cpu",
        teacher_forcing_ratio=0.5,
        logger=None,
    ):
        """Store the training components and move models to the chosen device."""
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.logger = logger

    def train_epoch(self, epoch_number=None, total_epochs=None):
        """Run one full pass over the dataloader and return the average loss."""
        self.encoder.train()
        self.decoder.train()

        epoch_loss = 0

        progress_bar = tqdm(
            self.dataloader,
            desc=f"Train {epoch_number}/{total_epochs}" if epoch_number is not None else "Train",
            leave=False,
        )

        for batch_index, (src, trg) in enumerate(progress_bar, start=1):
            # Move the current batch to the same device as the models.
            src = src.to(self.device)      # [src_len, batch]
            trg = trg.to(self.device)      # [trg_len, batch]

            self.optimizer.zero_grad()

            # Encode the full source sequence once before decoding token by token.
            encoder_outputs, hidden = self.encoder(src)
            # encoder_outputs: [src_len, batch, hidden*2]
            # hidden: [batch, hidden]

            # Start the decoder with the <sos> token from the target sequence.
            input_token = trg[0]  # [batch]

            loss = 0

            for t in range(1, trg.shape[0]):
                # Predict the next token using the previous token and attention context.
                output, hidden = self.decoder(
                    input_token,
                    hidden,
                    encoder_outputs
                )
                # output: [batch, vocab]

                # Accumulate token-level loss across the whole target sentence.
                loss += self.criterion(output, trg[t])

                # Sometimes feed the ground-truth token back in to stabilize training.
                teacher_force = random.random() < self.teacher_forcing_ratio
                top1 = output.argmax(1)

                input_token = trg[t] if teacher_force else top1

            # Backpropagate through the entire sequence loss.
            loss.backward()

            # Clip gradients to reduce the chance of exploding updates.
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1
            )

            self.optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{epoch_loss / batch_index:.4f}")

        return epoch_loss / len(self.dataloader)

    def _log(self, message):
        """Send messages to the configured logger or fall back to print."""
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _save_checkpoint(self, checkpoint_path, epoch, train_loss, metrics, extra_payload=None):
        """Persist the current best full model objects to disk."""
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_accuracy": metrics["accuracy"],
            "validation_bleu": metrics["bleu"],
            "encoder": self.encoder,
            "decoder": self.decoder,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

        if extra_payload is not None:
            checkpoint.update(extra_payload)

        torch.save(checkpoint, checkpoint_path)

    def train(self, epochs, evaluator=None, checkpoint_path=None, checkpoint_payload=None):
        """Train for multiple epochs and optionally evaluate after each epoch."""
        best_accuracy = float("-inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(
                epoch_number=epoch + 1,
                total_epochs=epochs,
            )

            if evaluator is None:
                self._log(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}")
                continue

            metrics = evaluator.evaluate_bleu(
                epoch_number=epoch + 1,
                total_epochs=epochs,
            )
            self._log(
                "Epoch "
                f"{epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Accuracy: {metrics['accuracy']:.4f} | "
                f"Val BLEU: {metrics['bleu']:.4f}"
            )

            if metrics["accuracy"] > best_accuracy:
                best_accuracy = metrics["accuracy"]

                if checkpoint_path is not None:
                    self._save_checkpoint(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch + 1,
                        train_loss=train_loss,
                        metrics=metrics,
                        extra_payload=checkpoint_payload,
                    )

                self._log(
                    f"New best model saved with validation accuracy {best_accuracy:.4f}"
                )
