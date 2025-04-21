
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import librosa
from torch.nn.utils.rnn import pad_sequence

# Define CNN-based audio denoising model
class AudioDenoiserCNN(nn.Module):
    def __init__(self):
        super(AudioDenoiserCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=15, stride=1, padding=7),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=15, stride=1, padding=7),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

