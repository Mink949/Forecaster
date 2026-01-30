import torch
import torch.nn as nn

class DLinear(nn.Module):
    """
    DLinear Model: Simple Linear Layers for Time Series Forecasting
    Decomposes series into Trend and Remainder, applies Linear layers to each.
    Ref: https://arxiv.org/pdf/2205.13504.pdf
    """
    def __init__(self, input_len, pred_len, enc_in):
        super(DLinear, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        
        # Decomposition (Moving Average)
        self.kernel_size = 25
        self.decomposition = SeriesDecomp(self.kernel_size)
        
        self.Linear_Seasonal = nn.Linear(self.input_len, self.pred_len)
        self.Linear_Trend = nn.Linear(self.input_len, self.pred_len)
        
    def forward(self, x):
        # x: [Batch, Input_Len, Channels]
        seasonal_init, trend_init = self.decomposition(x)
        
        # Permute to [Batch, Channels, Input_Len] for Linear layer over time
        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)
        
        seasonal_output = self.Linear_Seasonal(seasonal_init)
        trend_output = self.Linear_Trend(trend_init)
        
        x = seasonal_output + trend_output
        
        # Permute back to [Batch, Pred_Len, Channels]
        return x.permute(0, 2, 1)

class SeriesDecomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class MovingAvg(nn.Module):
    """
    Moving average block to highlight trend
    """
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Padding on the both ends of time dimension
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = x.permute(0, 2, 1)
        x = self.avg(x)
        x = x.permute(0, 2, 1)
        return x

class LSTMModel(nn.Module):
    """
    Standard LSTM for Time Series
    """
    def __init__(self, input_size, hidden_size, num_layers, output_len):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_len = output_len
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_len) # Predicting 1 output channel (Deaths) over Horizon
        # Or predict all channels? Usually we care about target.
        # But DLinear predicts all channels (enc_in).
        # We will design this to predict only Target (1 channel) for simplicity, or alignment.
        # Let's align: Predict only Target density.
        
    def forward(self, x):
        # x: [Batch, Len, Feat]
        out, _ = self.lstm(x)
        # Take last time step
        out = out[:, -1, :]
        out = self.fc(out) 
        # Output: [Batch, Horizon] -> Matches target shape
        return out.unsqueeze(-1) # [Batch, Horizon, 1] if expecting 3D, or squeeze if 2D.
