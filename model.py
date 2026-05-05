import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class PoseLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=256, num_layers=2, num_classes=15):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        packed = rnn_utils.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        out = self.fc(h_n[-1])
        return out
