import torch
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        train_loader,
        val_loader,
        num_epochs,
        save_path,
        log_dir="runs",
        run_name=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.save_path = save_path

        # Create a unique log directory for each run
        if run_name is None:
            import pytz
            from datetime import datetime

            tz = pytz.timezone("Asia/Taipei")
            run_name = datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = os.path.join(log_dir, run_name)

        # Initialize the SummaryWriter
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self):
        best_val_loss = float("inf")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            train_loss = self._train_one_epoch(epoch)
            val_loss = self._validate(epoch)

            end_time = time.time()
            epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

            print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
            print(f"\tTrain Loss: {train_loss:.3f}")
            print(f"\tVal. Loss: {val_loss:.3f}")

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.save_path)
                print(f"Model saved to {self.save_path}")

        # Close the SummaryWriter
        self.writer.close()

    def _train_one_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0

        for batch_idx, (data, labels) in enumerate(self.train_loader):
            data = data.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(data)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Log training loss to TensorBoard
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar("Loss/train_batch", batch_loss, global_step)

            if (batch_idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{self.num_epochs}], Step [{batch_idx+1}/{len(self.train_loader)}], Loss: {batch_loss:.4f}"
                )

        avg_epoch_loss = epoch_loss / len(self.train_loader)
        # Log average epoch loss
        self.writer.add_scalar("Loss/train_epoch", avg_epoch_loss, epoch)
        return avg_epoch_loss

    def _validate(self, epoch):
        self.model.eval()
        epoch_loss = 0

        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for data, labels in self.val_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(data)
                loss = self.criterion(outputs, labels)

                epoch_loss += loss.item()

                # Collect outputs and labels for metric computation
                all_outputs.append(outputs.cpu())
                all_labels.append(labels.cpu())

        avg_val_loss = epoch_loss / len(self.val_loader)

        # Concatenate all outputs and labels
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        # Apply sigmoid to outputs to get probabilities
        probs = torch.sigmoid(all_outputs)

        # Binarize outputs with a threshold (e.g., 0.5)
        preds = (probs >= 0.5).float()

        # Compute metrics
        precision, recall, f1 = self._compute_metrics(preds, all_labels)

        # Log metrics to TensorBoard
        self.writer.add_scalar("Loss/val_epoch", avg_val_loss, epoch)
        self.writer.add_scalar("Metrics/precision", precision, epoch)
        self.writer.add_scalar("Metrics/recall", recall, epoch)
        self.writer.add_scalar("Metrics/f1_score", f1, epoch)

        print(
            f"\tVal. Precision: {precision:.3f} | Val. Recall: {recall:.3f} | Val. F1-score: {f1:.3f}"
        )

        return avg_val_loss

    def _compute_metrics(self, preds, labels):
        # Convert tensors to numpy arrays
        preds = preds.numpy()
        labels = labels.numpy()

        # Compute metrics
        precision = precision_score(labels, preds, average="macro", zero_division=0)
        recall = recall_score(labels, preds, average="macro", zero_division=0)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)

        return precision, recall, f1

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs
