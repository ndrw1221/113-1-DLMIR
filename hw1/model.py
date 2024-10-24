import torch
import torch.nn as nn
from transformers import AutoModel, Wav2Vec2FeatureExtractor


class Model(nn.Module):
    def __init__(
        self, num_classes=9, sampling_rate=24000, finetune_strategy="classifier only"
    ):
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
        if finetune_strategy == "classifier only":
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        elif finetune_strategy == "partial":
            # Freeze all layers first
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

            # Unfreeze the last 6 encoder layers for fine-tuning
            # Assuming the model has 24 encoder layers, unfreeze layers 18 to 23
            for layer_num in range(18, 24):
                for param in self.feature_extractor.encoder.layers[
                    layer_num
                ].parameters():
                    param.requires_grad = True

        elif finetune_strategy == "full":
            pass
        else:
            raise ValueError("Invalid finetune strategy.")

        self.pooling = nn.AdaptiveAvgPool1d(1)  # Pool over the sequence length

        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # x is a batch of audio samples: shape (batch_size, audio_length)
        x = x.cpu()
        inputs = [audio.numpy() for audio in x]

        processed = self.processor(
            inputs, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        device = next(self.parameters()).device
        processed = processed.to(device)

        outputs = self.feature_extractor(**processed)
        x = outputs.last_hidden_state  # (batch_size, sequence_length, 1024)
        x = x.permute(0, 2, 1)  # (batch_size, 1024, sequence_length)
        x = self.pooling(x).squeeze(-1)  # (batch_size, 1024)

        x = self.classifier(x)  # Pass through enhanced classifier
        return x
