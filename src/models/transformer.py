import torch
import torch.nn as nn
import math

class TimeSeriesTransformer(nn.Module):
    """
    Transformer-based model for Weather-Health Forecasting.
    
    Architecture:
    1. Positional Encoding: Standard sinusoidal encoding for temporal position.
    2. Transformer Encoder: Capturing temporal interactions via self-attention.
    3. Multi-Task Head: Forecasting (Regression) + Surge Detection (Classification).
    
    Note: This uses standard positional encoding, not explicit DLNM-style lag basis.
    The 'lag awareness' comes from the sequential structure of the input window.
    """
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1, output_len=4, max_len=52):
        super(TimeSeriesTransformer, self).__init__()
        
        self.d_model = d_model
        
        # Input Projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional Encoding (Standard Time PE + Lag PE)
        # Here we just use standard PE for the sequence, but we could add Lag specific embeddings if we had non-sequential lags.
        # Since our input is a sequence [t-W, ..., t-1], position roughly equals lag.
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Heads
        # 1. Forecast Head: Flat projection from all time steps? Or just last?
        # Creating a summary vector from the encoder output.
        # Option A: Flatten and project.
        # Option B: Attention pooling.
        # Option C: Use last token.
        # We'll use Flatten for simplicity over short windows (W=8 or 12).
        
        # But wait, input length varies? No, fixed W.
        # Flatten size = W * d_model. Can be large if W is large.
        # Let's use Global Average Pooling over time.
        
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_len) # Predicts H steps ahead
        )
        
        self.surge_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_len), # Predicts Surge Probability for each step
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [Batch, Window, Features]
        
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        # Transformer Encode
        # memory: [Batch, Window, d_model]
        memory = self.transformer_encoder(x)
        
        # Global Average Pooling over the time dimension
        # Summary of the window context
        summary = torch.mean(memory, dim=1) # [Batch, d_model]
        
        # Heads
        forecast = self.forecast_head(summary) # [Batch, Horizon]
        surge_probs = self.surge_head(summary) # [Batch, Horizon]
        
        return forecast, surge_probs

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq_Len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
