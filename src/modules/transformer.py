import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input embeddings to provide the model
    with information about the relative or absolute position of the tokens in the sequence.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerStack(nn.Module):
    """
    A Transformer-based expert for time-series forecasting.
    This model uses a Transformer Encoder to process the input sequence.
    """
    def __init__(self, input_size: int, horizon: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.input_embedding = nn.Linear(1, d_model) # Embed each time step
        self.d_model = d_model
        self.decoder = nn.Linear(d_model * input_size, horizon)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len]

        Returns:
            output: Tensor, shape [batch_size, horizon]
        """
        # Add a feature dimension and embed
        src = self.input_embedding(src.unsqueeze(-1)) * math.sqrt(self.d_model) # [B, T, D]

        # Add positional encoding
        # The PositionalEncoding expects [seq_len, batch_size, embedding_dim], so we permute
        src = self.pos_encoder(src.permute(1, 0, 2)).permute(1, 0, 2)

        # Pass through transformer encoder
        output = self.transformer_encoder(src) # [B, T, D]

        # Flatten and decode to horizon
        output = output.reshape(output.size(0), -1) # [B, T*D]
        output = self.decoder(output) # [B, H]

        return output
