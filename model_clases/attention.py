from time import sleep
import torch
from torch.autograd import backward
import torch.nn as nn


class GRUCell(nn.Module):
    """Implement a single custom GRU cell step."""

    def __init__(self, input_dim, hidden_dim) -> None:
        """Create the learnable projections used by the GRU gates."""
        super().__init__()

        self.hidden_dim = hidden_dim
        
        # The update gate decides how much new information to write.
        self.Wz = nn.Linear(input_dim, hidden_dim)
        self.Uz = nn.Linear(hidden_dim, hidden_dim)
        
        # The reset gate controls how much of the old state to forget.
        self.Wr = nn.Linear(input_dim, hidden_dim)
        self.Ur = nn.Linear(hidden_dim, hidden_dim)
        
        # The candidate state proposes the next hidden representation.
        self.Wh = nn.Linear(input_dim, hidden_dim)
        self.Uh = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, h_prev):
        """Update the hidden state for one time step."""
        # x: [batch, input_dim]
        # h_prev: [batch, hidden_dim]
        z = torch.sigmoid(self.Wz(x) + self.Uz(h_prev))
        r = torch.sigmoid(self.Wr(x) + self.Ur(h_prev))

        # Blend the reset state into the candidate before mixing it with h_prev.
        h_tilde = torch.tanh(self.Wh(x) + self.Uh(r*h_prev))

        h = (1-z) * h_prev + z * h_tilde

        return h

class GRULayer(nn.Module):
    """Roll the custom GRU cell over a full sequence."""

    def __init__(self, input_dim, hidden_dim) -> None:
        """Create one recurrent layer backed by the custom GRU cell."""
        super().__init__()
        self.cell = GRUCell(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        """Process a sequence and return all hidden states plus the final one."""
        # x: [seq_len, batch, input_dim]
        seq_len, batch_size, _ = x.shape

        # Start from an all-zero hidden state for each sequence in the batch.
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []

        for t in range(seq_len):
            # Feed one time step at a time through the recurrent cell.
            h = self.cell(x[t], h)
            outputs.append(h.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)

        return outputs, h

class Encoder(nn.Module):
    """Encode a source sentence with forward and backward recurrent passes."""

    def __init__(self, input_dim, emb_dim, hidden_dim):
        """Create embeddings plus separate recurrent passes for both directions."""
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.forward_gru = GRULayer(emb_dim, hidden_dim)
        self.backward_gru = GRULayer(emb_dim, hidden_dim)

    def forward(self, src):
        """Embed the source tokens and run the bidirectional encoder."""
        # src: [seq_len, batch]
        embedded = self.embedding(src)
        # [seq_len, batch, emb_dim]
        
        # Read the sentence from left to right.
        forward_outputs, forward_hidden = self.forward_gru(embedded)

        # Read the same sentence from right to left for extra context.
        reversed_embedded = torch.flip(embedded, dims=[0])
        backward_outputs, backward_hidden = self.backward_gru(reversed_embedded)

        # Flip the backward outputs back into the original time order.
        backward_outputs = torch.flip(backward_outputs, dims=[0])

        # Join both directional representations for attention.
        encoder_outputs = torch.cat((forward_outputs, backward_outputs), dim=2)

        # Merge the final states into one decoder-sized hidden representation.
        hidden = torch.tanh(forward_hidden + backward_hidden)

        return encoder_outputs, hidden

class Attention(nn.Module):
    """Score how strongly the decoder should focus on each encoder step."""

    def __init__(self, hidden_dim):
        """Create the small feed-forward network used for attention scoring."""
        super().__init__()

        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        """Return normalized attention weights over the encoder sequence."""
        # hidden: [batch, hidden_dim]
        # encoder_outputs: [seq_len, batch, hidden_dim*2]

        seq_len = encoder_outputs.shape[0]

        # Repeat the decoder state so it can be compared with every source step.
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        # [batch, seq_len, hidden_dim]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch, seq_len, hidden_dim*2]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        )

        attention = self.v(energy).squeeze(2)
        # [batch, seq_len]

        return torch.softmax(attention, dim=1)

class Decoder(nn.Module):
    """Decode one target token at a time with attention over the encoder states."""

    def __init__(self, output_dim, emb_dim, hidden_dim, attention):
        """Create the embedding, recurrent, attention, and output layers."""
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # The decoder consumes both the token embedding and the attention context.
        self.gru_cell = GRUCell(
            input_dim=emb_dim + hidden_dim * 2,
            hidden_dim=hidden_dim
        )

        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2, output_dim)

        self.attention = attention

    def forward(self, input_token, hidden, encoder_outputs, return_attention=False):
        """Predict the next token distribution and updated decoder state."""
        # input_token: [batch]
        # hidden: [batch, hidden_dim]

        input_token = input_token.unsqueeze(0)
        # [1, batch]

        embedded = self.embedding(input_token).squeeze(0)
        # [batch, emb_dim]

        # Measure which encoder states matter most for this decoding step.
        attn_weights = self.attention(hidden, encoder_outputs)
        # [batch, seq_len]

        attn_weights = attn_weights.unsqueeze(1)
        # [batch, 1, seq_len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # [batch, seq_len, hidden_dim*2]

        context = torch.bmm(attn_weights, encoder_outputs)
        # [batch, 1, hidden_dim*2]

        context = context.squeeze(1)
        # [batch, hidden_dim*2]

        # Join the current token embedding with the context from the encoder.
        gru_input = torch.cat((embedded, context), dim=1)

        hidden = self.gru_cell(gru_input, hidden)
        # [batch, hidden_dim]

        # Project the decoder state into vocabulary logits.
        output = self.fc_out(torch.cat((hidden, context), dim=1))
        # [batch, vocab_size]

        if return_attention:
            return output, hidden, attn_weights.squeeze(1)

        return output, hidden
