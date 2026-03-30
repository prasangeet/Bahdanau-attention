from pathlib import Path

import matplotlib.pyplot as plt


def save_attention_plot(source_tokens, target_tokens, attention_matrix, output_path, title):
    """Save a heatmap that shows which source words the decoder attended to."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not source_tokens:
        source_tokens = ["<empty>"]

    if not target_tokens:
        target_tokens = ["<empty>"]

    if len(attention_matrix) == 0:
        attention_matrix = [[0.0 for _ in source_tokens]]

    figure, axis = plt.subplots(figsize=(max(8, len(source_tokens) * 0.6), max(6, len(target_tokens) * 0.5)))
    image = axis.imshow(attention_matrix, aspect="auto", cmap="gray")

    axis.set_xticks(range(len(source_tokens)))
    axis.set_xticklabels(source_tokens, rotation=45, ha="right")
    axis.set_yticks(range(len(target_tokens)))
    axis.set_yticklabels(target_tokens)
    axis.set_xlabel("Source Tokens")
    axis.set_ylabel("Predicted Tokens")
    axis.set_title(title)

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
