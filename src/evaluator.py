import torch
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm


class Evaluator:
    """Run greedy decoding on a dataloader and report validation metrics."""

    def __init__(
        self,
        encoder,
        decoder,
        dataloader,
        target_idx2word,
        sos_idx,
        eos_idx,
        pad_idx,
        device="cpu",
    ):
        """Store the model pieces and the metadata needed to decode token ids."""
        self.encoder = encoder
        self.decoder = decoder
        self.dataloader = dataloader
        self.target_idx2word = target_idx2word
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.device = device

    def decode_batch(self, src, max_len=50, return_attention=False):
        """Greedily decode a batch and optionally keep the attention weights."""
        src = src.to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        batch_size = src.shape[1]
        input_token = torch.full(
            (batch_size,),
            self.sos_idx,
            dtype=torch.long,
            device=self.device,
        )

        generated_tokens = [[] for _ in range(batch_size)]
        attention_steps = []

        for _ in range(max_len):
            if return_attention:
                output, hidden, attention = self.decoder(
                    input_token,
                    hidden,
                    encoder_outputs,
                    return_attention=True,
                )
                attention_steps.append(attention.detach().cpu())
            else:
                output, hidden = self.decoder(input_token, hidden, encoder_outputs)

            top1 = output.argmax(1)

            for batch_index in range(batch_size):
                generated_tokens[batch_index].append(top1[batch_index].item())

            input_token = top1

        return generated_tokens, attention_steps

    def _tokens_to_words(self, token_ids):
        """Convert token ids into words while skipping special tokens."""
        words = []

        for token_id in token_ids:
            if token_id in {self.pad_idx, self.sos_idx}:
                continue

            if token_id == self.eos_idx:
                break

            words.append(self.target_idx2word.get(token_id, "<unk>"))

        return words

    def decode_single_example(self, src_tokens, max_len=50):
        """Decode one source sequence and return predicted words plus attention."""
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            src_tensor = torch.tensor(src_tokens, dtype=torch.long).unsqueeze(1)
            generated_tokens, attention_steps = self.decode_batch(
                src_tensor,
                max_len=max_len,
                return_attention=True,
            )

        predicted_token_ids = generated_tokens[0]
        predicted_words = self._tokens_to_words(predicted_token_ids)

        if attention_steps:
            attention_matrix = torch.stack(attention_steps).squeeze(1).numpy()
            attention_matrix = attention_matrix[: len(predicted_words), : len(src_tokens)]
        else:
            attention_matrix = np.zeros((0, len(src_tokens)))

        return predicted_words, attention_matrix

    def evaluate_bleu(self, max_len=50, epoch_number=None, total_epochs=None, split_name="Validate"):
        """Decode the evaluation set and compute BLEU plus token accuracy."""
        self.encoder.eval()
        self.decoder.eval()

        references = []
        hypotheses = []
        smoothing = SmoothingFunction().method1
        correct_tokens = 0
        total_tokens = 0

        with torch.no_grad():
            progress_bar = tqdm(
                self.dataloader,
                desc=(
                    f"{split_name} {epoch_number}/{total_epochs}"
                    if epoch_number is not None
                    else split_name
                ),
                leave=False,
            )

            for src, trg in progress_bar:
                # Move the evaluation batch onto the same device as the models.
                trg = trg.to(self.device)
                generated_tokens, _ = self.decode_batch(src, max_len=max_len)
                batch_size = trg.shape[1]

                # Measure token accuracy against the ground-truth target sequence.
                target_len = trg.shape[0] - 1
                predictions = torch.tensor(
                    generated_tokens,
                    dtype=torch.long,
                    device=self.device,
                ).transpose(0, 1)
                predictions = predictions[:target_len]
                target_tokens = trg[1 : 1 + predictions.shape[0]]
                valid_mask = target_tokens != self.pad_idx

                correct_tokens += (
                    (predictions == target_tokens) & valid_mask
                ).sum().item()
                total_tokens += valid_mask.sum().item()

                # Compare each prediction with its matching reference sentence.
                for batch_index in range(batch_size):
                    reference_ids = trg[:, batch_index].tolist()
                    hypothesis_ids = generated_tokens[batch_index]

                    reference_words = self._tokens_to_words(reference_ids)
                    hypothesis_words = self._tokens_to_words(hypothesis_ids)

                    references.append([reference_words])
                    hypotheses.append(hypothesis_words)

                accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
                progress_bar.set_postfix(acc=f"{accuracy:.4f}")

        bleu = corpus_bleu(references, hypotheses, smoothing_function=smoothing)
        accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0

        return {
            "bleu": bleu,
            "accuracy": accuracy,
        }
