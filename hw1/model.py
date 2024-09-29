import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, Wav2Vec2FeatureExtractor
import pdb


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

        # Pooling layer (Mean pooling)
        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pooling over the sequence length

        # Fully connected layer to map embeddings to the number of classes
        self.fc = nn.Linear(1024, num_classes)

        # Softmax for class probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Preprocess the audio data using the processor
        x = self.processor(x, sampling_rate=self.sampling_rate, return_tensors="pt")

        # Perform a forward pass using the model
        x = self.feature_extractor(**x)

        # Get the last hidden state (shape: batch_size, sequence_length, embedding_dim)
        x = x.last_hidden_state  # (1, 375, 1024)

        # Permute to shape (batch_size, embedding_dim, sequence_length)
        x = x.permute(0, 2, 1)  # (1, 1024, 375)

        # Apply pooling to reduce the sequence length
        x = self.pooling(x).squeeze(-1)  # (1, 1024)

        # Pass through the fully connected layer
        x = self.fc(x)  # (1, num_classes)

        # Apply Softmax to get class probabilities
        x = self.softmax(x)  # (1, num_classes)

        return x


# Example usage
model = Model()
input_audio_path = "/home/ndrw1221/nas/slakh-dataset(dlmir-hw1)/test/Track01876_13.npy"
input_audio = np.load(input_audio_path)
with torch.no_grad():
    output = model(input_audio)
pdb.set_trace()
# all_layer_hidden_states = torch.stack(output.hidden_states).squeeze()
# print(all_layer_hidden_states.shape)
