import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Encode the source sequence with PyTorch's built-in bidirectional GRU."""

    def __init__(self, input_dim, emb_dim, hidden_dim):
        """Create the embedding layer and the bidirectional encoder GRU."""
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            bidirectional=True,
        )

    def forward(self, src):
        """Return encoder outputs plus a decoder-sized hidden state."""
        # src: [seq_len, batch]
        embedded = self.embedding(src)
        # [seq_len, batch, emb_dim]

        # The bidirectional GRU returns both time-step outputs and final states.
        encoder_outputs, hidden = self.gru(embedded)
        # encoder_outputs: [seq_len, batch, hidden_dim * 2]
        # hidden: [2, batch, hidden_dim]

        # Combine the final forward and backward states into one decoder state.
        hidden = torch.tanh(hidden[0] + hidden[1])
        # [batch, hidden_dim]

        return encoder_outputs, hidden


class Attention(nn.Module):
    """Compute attention weights over the encoder outputs."""

    def __init__(self, hidden_dim):
        """Create the projection layers used to score source positions."""
        super().__init__()

        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """Return normalized attention scores for each source token."""
        # hidden: [batch, hidden_dim]
        # encoder_outputs: [seq_len, batch, hidden_dim * 2]
        seq_len = encoder_outputs.shape[0]

        # Repeat the decoder state across time so each source step can be scored.
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # [batch, seq_len, hidden_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch, seq_len, hidden_dim * 2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )

        attention = self.v(energy).squeeze(2)
        # [batch, seq_len]

        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    """Decode one token at a time with attention and PyTorch's built-in GRU."""

    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        """Create the embedding, attention-aware GRU, and output projection."""
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.gru = nn.GRU(
            input_size=emb_dim + hidden_dim * 2,
            hidden_size=hidden_dim,
        )
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2, output_dim)

    def forward(self, input_token, hidden, encoder_outputs, return_attention=False):
        """Predict the next token distribution and the next hidden state."""
        # input_token: [batch]
        # hidden: [batch, hidden_dim]
        input_token = input_token.unsqueeze(0)
        # [1, batch]

        embedded = self.embedding(input_token)
        # [1, batch, emb_dim]

        # Use attention to pull the most relevant source context for this step.
        attn_weights = self.attention(hidden, encoder_outputs)
        # [batch, seq_len]

        attn_weights = attn_weights.unsqueeze(1)
        # [batch, 1, seq_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch, seq_len, hidden_dim * 2]

        context = torch.bmm(attn_weights, encoder_outputs)
        # [batch, 1, hidden_dim * 2]

        context = context.permute(1, 0, 2)
        # [1, batch, hidden_dim * 2]

        # Concatenate the current embedding with the attended encoder context.
        gru_input = torch.cat((embedded, context), dim=2)
        # [1, batch, emb_dim + hidden_dim * 2]

        output, hidden = self.gru(gru_input, hidden.unsqueeze(0))
        # output: [1, batch, hidden_dim]
        # hidden: [1, batch, hidden_dim]

        output = output.squeeze(0)
        hidden = hidden.squeeze(0)
        context = context.squeeze(0)

        # Map the decoder state and context into vocabulary logits.
        prediction = self.fc_out(torch.cat((output, context), dim=1))
        # [batch, output_dim]

        if return_attention:
            return prediction, hidden, attn_weights.squeeze(1)

        return prediction, hidden
