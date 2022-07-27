from typing import Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model : int, max_len : int = 128):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model).float()
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)

        self.register_buffer('pe', pe)

    def forward(self, x:torch.Tensor):
        return x + self.pe[:x.size(0), :]

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class TStransformer(nn.Module):
    def __init__(self, feature_dims : int = 256, max_len : int = 128, n_layers : int = 1, n_heads : int = 8, dropout : float = 0.1, cls_dims : int = 128, n_classes : int = 2):
        super(TStransformer, self).__init__()
        self.src_mask = None
        self.pos_enc = PositionalEncoding(d_model = feature_dims, max_len = max_len)
        self.encoder = nn.TransformerEncoderLayer(d_model = feature_dims, nhead=n_heads, dropout = dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dims, cls_dims),
            nn.BatchNorm1d(cls_dims),
            nn.ReLU(),
            nn.Linear(cls_dims, n_classes)
        )

    def forward(self, x : torch.Tensor)->torch.Tensor:
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        
        x = self.pos_enc(x)
        x = self.transformer_encoder(x, self.src_mask).permute(1,0,2).mean(dim = 1) # (seq_len, batch, feature_dims)
        x = self.classifier(x)
    
        return x

    def _generate_square_subsequent_mask(self, size : int)->torch.Tensor:
        mask = (torch.triu(torch.ones(size,size))==1).transpose(0,1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == "__main__":
    # test
    time        = np.arange(0, 400, 0.1)
    amplitude   = np.sin(time) + np.sin(time*0.05) +np.sin(time*0.12) *np.random.normal(-0.2, 0.2, len(time))

    input_window = 128
    output_window = 1
    batch_size = 10 # batch size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TStransformer(
        feature_dims = 250,
        max_len = 5000,
        n_layers = 1,
        n_heads = 10,
        dropout = 0.5
    )

    model.to(device)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) 
    amplitude = scaler.fit_transform(amplitude.reshape(-1, 1)).reshape(-1)

    sampels = 2600
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    def create_inout_sequences(input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = np.append(input_data[i:i+tw][:-output_window] , output_window * [0])
            train_label = input_data[i:i+tw]
            #train_label = input_data[i+output_window:i+tw+output_window]
            inout_seq.append((train_seq ,train_label))
        return torch.FloatTensor(inout_seq)

    def get_batch(source, i, batch_size, input_window):
        seq_len = min(batch_size, len(source) - 1 - i)
        data = source[i:i+seq_len]    
        input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
        target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
        return input, target

    train_sequence = create_inout_sequences(train_data, input_window).to(device)
    train_sequence = train_sequence[:-output_window] #todo: fix hack?

    sample_data, sample_target = get_batch(train_sequence, 0, batch_size = batch_size, input_window=input_window)
    sample_output = model(sample_data)

    print("sample_data : ", sample_data.size())
    print("sample_target : ", sample_target.size())
    print("sample_output : ", sample_output.size())