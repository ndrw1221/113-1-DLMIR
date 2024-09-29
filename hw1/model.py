# model.py

import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class Model(nn.Module):
    def __init__(self, num_classes=9, sampling_rate=24000):
        super(Model, self).__init__()
        # Load the pre-trained model and processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.feature_extractor = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.sampling_rate = sampling_rate

        # Freeze the pre-trained model's parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool over the sequence length

        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # x is a batch of audio samples: shape (batch_size, audio_length)

        # Ensure x is on CPU before converting to numpy arrays
        x = x.cpu()

        # Convert tensors to lists of numpy arrays
        inputs = [audio.numpy() for audio in x]

        # Preprocess the audio data using the processor
        processed = self.processor(
            inputs, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )

        # Move processed inputs to the same device as the model
        device = next(self.parameters()).device
        processed = processed.to(device)

        # Pass through the pre-trained feature extractor
        outputs = self.feature_extractor(**processed)

        x = outputs.last_hidden_state  # (batch_size, sequence_length, 1024)

        x = x.permute(0, 2, 1)  # (batch_size, 1024, sequence_length)

        x = self.pooling(x).squeeze(-1)  # (batch_size, 1024)

        x = self.fc(x)  # (batch_size, num_classes)

        return x
