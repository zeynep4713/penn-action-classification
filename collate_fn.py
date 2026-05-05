import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    sequences, lengths, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)  # (batch, max_len, 51)
    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels)
    return sequences_padded, lengths, labels
