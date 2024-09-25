# coding: utf-8
import os
import time
import numpy as np
import pandas as pd
import datetime
import tqdm
import csv
import fire
import argparse
import pickle
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.autograd import Variable

# from solver import skip_files
from sklearn.preprocessing import LabelBinarizer

import model as Model

skip_files = set([])
TAGS = [
    "genre---downtempo",
    "genre---ambient",
    "genre---rock",
    "instrument---synthesizer",
    "genre---atmospheric",
    "genre---indie",
    "instrument---electricpiano",
    "genre---newage",
    "instrument---strings",
    "instrument---drums",
    "instrument---drummachine",
    "genre---techno",
    "instrument---guitar",
    "genre---alternative",
    "genre---easylistening",
    "genre---instrumentalpop",
    "genre---chillout",
    "genre---metal",
    "mood/theme---happy",
    "genre---lounge",
    "genre---reggae",
    "genre---popfolk",
    "genre---orchestral",
    "instrument---acousticguitar",
    "genre---poprock",
    "instrument---piano",
    "genre---trance",
    "genre---dance",
    "instrument---electricguitar",
    "genre---soundtrack",
    "genre---house",
    "genre---hiphop",
    "genre---classical",
    "mood/theme---energetic",
    "genre---electronic",
    "genre---world",
    "genre---experimental",
    "instrument---violin",
    "genre---folk",
    "mood/theme---emotional",
    "instrument---voice",
    "instrument---keyboard",
    "genre---pop",
    "instrument---bass",
    "instrument---computer",
    "mood/theme---film",
    "genre---triphop",
    "genre---jazz",
    "genre---funk",
    "mood/theme---relaxing",
]


def read_file(tsv_file):
    tracks = {}
    with open(tsv_file) as fp:
        reader = csv.reader(fp, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            track_id = row[0]
            tracks[track_id] = {
                "path": row[3].replace(".mp3", ".npy"),
                "tags": row[5:],
            }
    return tracks


class Predict(object):
    def __init__(self, config):
        self.model_type = config.model_type
        self.model_load_path = config.model_load_path
        self.dataset = config.dataset
        self.data_path = config.data_path
        self.batch_size = config.batch_size
        self.is_cuda = torch.cuda.is_available()
        self.build_model()
        self.get_dataset()

    def get_model(self):
        if self.model_type == "fcn":
            self.input_length = 29 * 16000
            return Model.FCN()
        elif self.model_type == "musicnn":
            self.input_length = 3 * 16000
            return Model.Musicnn(dataset=self.dataset)
        elif self.model_type == "crnn":
            self.input_length = 29 * 16000
            return Model.CRNN()
        elif self.model_type == "sample":
            self.input_length = 59049
            return Model.SampleCNN()
        elif self.model_type == "se":
            self.input_length = 59049
            return Model.SampleCNNSE()
        elif self.model_type == "attention":
            self.input_length = 15 * 16000
            return Model.CNNSA()
        elif self.model_type == "hcnn":
            self.input_length = 5 * 16000
            return Model.HarmonicCNN()
        elif self.model_type == "short":
            self.input_length = 59049
            return Model.ShortChunkCNN()
        elif self.model_type == "short_res":
            self.input_length = 59049
            return Model.ShortChunkCNN_Res()
        else:
            print(
                "model_type has to be one of [fcn, musicnn, crnn, sample, se, short, short_res, attention]"
            )

    def build_model(self):
        self.model = self.get_model()

        # load model
        self.load(self.model_load_path)

        # cuda
        if self.is_cuda:
            self.model.cuda()

    def get_dataset(self):
        if self.dataset == "mtat":
            self.test_list = np.load("./../split/mtat/test.npy")
            self.binary = np.load("./../split/mtat/binary.npy")
        if self.dataset == "msd":
            test_file = os.path.join("./../split/msd", "filtered_list_test.cP")
            test_list = pickle.load(open(test_file, "rb"), encoding="bytes")
            self.test_list = [
                value for value in test_list if value.decode() not in skip_files
            ]
            id2tag_file = os.path.join("./../split/msd", "msd_id_to_tag_vector.cP")
            self.id2tag = pickle.load(open(id2tag_file, "rb"), encoding="bytes")
        if self.dataset == "jamendo":
            test_file = os.path.join(
                "./../split/mtg-jamendo", "autotagging_top50tags-test.tsv"
            )
            self.file_dict = read_file(test_file)
            self.test_list = list(self.file_dict.keys())
            self.mlb = LabelBinarizer().fit(TAGS)
        if self.dataset == "nsynth":
            self.test_list = np.load("./../split/nsynth/test.npy")
            self.label = np.load("./data/nsynth/test/label.npy")

    def load(self, filename):
        S = torch.load(filename)
        if "spec.mel_scale.fb" in S.keys():
            self.model.spec.mel_scale.fb = S["spec.mel_scale.fb"]
        self.model.load_state_dict(S)

    def to_var(self, x):
        # if torch.cuda.is_available():
        #     x = x.cuda()
        # return Variable(x)
        if self.is_cuda:
            x = x.cuda()
        return x

    def get_tensor(self, fn):
        # load audio
        if self.dataset == "mtat":
            npy_path = (
                os.path.join(self.data_path, "mtat", "npy", fn.split("/")[1][:-3])
                + "npy"
            )
        elif self.dataset == "msd":
            msid = fn.decode()
            filename = "{}/{}/{}/{}.npy".format(msid[2], msid[3], msid[4], msid)
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset == "jamendo":
            filename = self.file_dict[fn]["path"]
            npy_path = os.path.join(self.data_path, filename)
        elif self.dataset == "nsynth":
            npy_path = (
                os.path.join(self.data_path, "nsynth", "npy", fn.split("/")[-1][:-3])
                + "npy"
            )
        raw = np.load(npy_path, mmap_mode="r")

        # split chunk
        # length = len(raw)
        # hop = (length - self.input_length) // self.batch_size
        # x = torch.zeros(self.batch_size, self.input_length)
        # for i in range(self.batch_size):
        #     x[i] = torch.Tensor(raw[i*hop:i*hop+self.input_length]).unsqueeze(0)
        # return x
        # Ensure the audio length is sufficient
        if len(raw) < self.input_length:
            raw = np.pad(raw, (0, self.input_length - len(raw)), "constant")
        # Randomly sample a chunk
        random_idx = np.random.randint(0, len(raw) - self.input_length + 1)
        npy = raw[random_idx : random_idx + self.input_length]
        return npy  # Shape: (input_length,)

    # def get_auc(self, est_array, gt_array):
    #     roc_aucs  = metrics.roc_auc_score(gt_array, est_array, average='macro')
    #     pr_aucs = metrics.average_precision_score(gt_array, est_array, average='macro')
    #     return roc_aucs, pr_aucs

    def test(self):
        # roc_auc, pr_auc, loss = self.get_test_score()
        # print('loss: %.4f' % loss)
        # print('roc_auc: %.4f' % roc_auc)
        # print('pr_auc: %.4f' % pr_auc)
        top1_accuracy, top3_accuracy, confusion_mat, loss = self.get_test_score()
        print(f"Test Loss: {loss:.4f}")
        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-3 Accuracy: {top3_accuracy:.4f}")
        print("Confusion Matrix:")
        print(confusion_mat)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title("Confusion Matrix on Test Set")
        plt.savefig("confusion_matrix_to_db_1.png")

    def get_test_score(self):
        self.model = self.model.eval()
        est_array = []
        gt_array = []
        losses = []
        # reconst_loss = nn.BCELoss()
        reconst_loss = nn.CrossEntropyLoss()
        top3_correct = 0
        total_samples = 0
        index = 0
        with torch.no_grad():
            for line in tqdm.tqdm(self.test_list):
                if self.dataset == "mtat":
                    ix, fn = line.split("\t")
                elif self.dataset == "msd":
                    fn = line
                    if fn.decode() in skip_files:
                        continue
                elif self.dataset == "jamendo":
                    fn = line
                elif self.dataset == "nsynth":
                    ix, fn = line.split("\t")

                # load and split
                x = self.get_tensor(fn)

                # ground truth
                if self.dataset == "mtat":
                    ground_truth = self.binary[int(ix)]
                elif self.dataset == "msd":
                    ground_truth = self.id2tag[fn].flatten()
                elif self.dataset == "jamendo":
                    ground_truth = np.sum(
                        self.mlb.transform(self.file_dict[fn]["tags"]), axis=0
                    )
                elif self.dataset == "nsynth":
                    ground_truth = self.label[int(ix)]

                # forward
                x = torch.tensor(x).unsqueeze(0).float()
                x = self.to_var(x)
                # y = torch.tensor([ground_truth.astype('float32') for i in range(self.batch_size)]).cuda()
                y = torch.tensor([ground_truth]).to(x.device).long()
                out = self.model(x)
                loss = reconst_loss(out, y)
                losses.append(loss.item())
                # out = out.detach().cpu()

                # estimate
                # estimated = np.array(out).mean(axis=0)
                # est_array.append(estimated)
                # gt_array.append(ground_truth)

                # Predictions
                probs = nn.functional.softmax(out, dim=1)
                top1_prob, top1_pred = torch.max(probs, dim=1)
                top3_probs, top3_preds = torch.topk(probs, k=3, dim=1)

                # Collect Top-1 predictions
                est_array.extend(top1_pred.cpu().numpy())
                gt_array.extend(y.cpu().numpy())

                # Compute Top-3 accuracy
                total_samples += 1
                if ground_truth in top3_preds.cpu().numpy()[0]:
                    top3_correct += 1

                index += 1

        # est_array, gt_array = np.array(est_array), np.array(gt_array)
        # loss = np.mean(losses)

        # roc_auc, pr_auc = self.get_auc(est_array, gt_array)
        # return roc_auc, pr_auc, loss

        # Compute average loss
        loss = np.mean(losses)
        # Compute Top-1 accuracy
        top1_accuracy = metrics.accuracy_score(gt_array, est_array)
        # Compute Top-3 accuracy
        top3_accuracy = top3_correct / total_samples
        # Compute confusion matrix
        confusion_mat = metrics.confusion_matrix(gt_array, est_array)

        return top1_accuracy, top3_accuracy, confusion_mat, loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        default="mtat",
        choices=["mtat", "msd", "jamendo", "nsynth"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="fcn",
        choices=[
            "fcn",
            "musicnn",
            "crnn",
            "sample",
            "se",
            "short",
            "short_res",
            "attention",
            "hcnn",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model_load_path", type=str, default=".")
    parser.add_argument("--data_path", type=str, default="./data")

    config = parser.parse_args()

    p = Predict(config)
    p.test()
