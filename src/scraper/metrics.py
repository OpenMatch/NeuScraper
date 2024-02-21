import torch
from torch import log, mean, round, sum
import numpy as np
import json
import math

eps = 1e-7

classes_weight_map = {
    0: ["Primary", 1.0, 1.0, 1.0],
    1: ["Heading", 1.0, 1.0, 1.0],
    2: ["Title", 1.0, 1.0, 1.0],
    3: ["Paragraph", 1.0, 1.0, 1.0],
    4: ["Table", 1.0, 1.0, 1.0],
    5: ["List", 1.0, 1.0, 1.0],
}


class ContentExtractionLoss:

    def __init__(self):
        self.classes_weight_map = classes_weight_map

    def get_weight_map(self):
        return self.classes_weight_map

    def load_weight_map(self, weight_map_json_path):
        with open(weight_map_json_path, "r") as f:
            weight_map_loaded = json.load(f)

        for k in self.classes_weight_map:
            self.classes_weight_map[k] = weight_map_loaded[str(k)]

    def single_class_weighted_crossentropy(self, y_true, y_pred, pos_weight, neg_weight):
        pos_loss = y_true * log(y_pred) * pos_weight
        neg_loss = (1 - y_true) * log(1 - y_pred) * neg_weight
        logloss = -(pos_loss + neg_loss)
        return mean(logloss, -1), pos_loss, neg_loss

    def weighted_crossentropy(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, min=eps, max=(1.0 - eps))
        y_true = torch.clamp(y_true, min=eps, max=(1.0 - eps))
        num_classes = y_pred.size()[2]

        loss = 0.0
        class_loss_list = torch.zeros(2, num_classes)

        for i in range(num_classes):
            positive_weight = self.classes_weight_map[i][1]
            negative_weight = self.classes_weight_map[i][2]
            class_weight = self.classes_weight_map[i][3]

            weighted_ce_loss, pos_loss, neg_loss = self.single_class_weighted_crossentropy(
                y_true[:, :, i], y_pred[:, :, i], positive_weight, negative_weight
            )
            class_loss_list[0][i] = pos_loss.mean()
            class_loss_list[1][i] = neg_loss.mean()

            if not math.isfinite(neg_loss.mean().item()):
                torch.set_printoptions(profile="full")
                print(self.classes_weight_map[i][0] + " encounter infinity loss!!!!!!!!\n")
                nan_in_neg_loss = torch.isnan(neg_loss)
                print("Any nan in neg loss: \n")
                print(nan_in_neg_loss[nan_in_neg_loss])
                inf_in_neg_loss = torch.isinf(neg_loss)
                print("Any inf in neg loss: \n")
                print(inf_in_neg_loss[inf_in_neg_loss])
                filter_invalid_loss = inf_in_neg_loss | nan_in_neg_loss

                print("invalid negative loss: \n")
                print(neg_loss[filter_invalid_loss])
                print("invalid Y predict \n")
                print(y_pred[:, :, i][filter_invalid_loss])
                print("y_true: \n")
                print(y_true[:, :, i][filter_invalid_loss])
                torch.set_printoptions(profile="default")
            loss += class_weight * weighted_ce_loss

        return loss.mean(), class_loss_list

    def precision_recall_evaluation(self, y_true, y_pred):
        result = {}
        num_classes = y_pred.size()[2]

        true_positives = torch.sum(y_true * round(y_pred),  (0, 1))
        possible_positives = torch.sum(y_true, (0, 1))
        predicted_positives = torch.sum(round(y_pred), (0, 1))

        precision = true_positives / (predicted_positives + eps)
        recall = true_positives / (possible_positives + eps)

        true_positives = true_positives.int().cpu().detach().tolist()
        possible_positives = possible_positives.int().cpu().detach().tolist()
        predicted_positives = predicted_positives.int().cpu().detach().tolist()
        precision = precision.cpu().detach().tolist()
        recall = recall.cpu().detach().tolist()

        for i in range(num_classes):
            class_name = self.classes_weight_map[i][0]
            result[class_name] = [precision[i], recall[i], true_positives[i], possible_positives[i], predicted_positives[i]]
        return result

    def merge_metrics_pr_result(self, result_list):
        tmp_result = None
        for result in result_list:
            if tmp_result is None:
                tmp_result = result
            else:
                for key in tmp_result:
                    if key in result:
                        tmp_result[key][2] += result[key][2]
                        tmp_result[key][3] += result[key][3]
                        tmp_result[key][4] += result[key][4]

                        tmp_result[key][0] = tmp_result[key][2] / (tmp_result[key][4] + eps)
                        tmp_result[key][1] = tmp_result[key][2] / (tmp_result[key][3] + eps)

        return tmp_result

    def print_result(self, result):
        if result is None or len(result) == 0:
            print("Evaluation result is none or contains 0 record")
            return

        print(
            "\nAnnotation".ljust(25, " ")
            + "Precision".ljust(15)
            + "Recall".ljust(12)
            + "True_Prediction".ljust(20)
            + "Possible Positive".ljust(20)
            + "Total_Prediction".ljust(20)
        )
        print("\n")

        for key in result:
            print(
                key.ljust(25, " ")
                + str(np.round(result[key][0], 4)).ljust(15)
                + str(np.round(result[key][1], 4)).ljust(12)
                + str(int(result[key][2])).ljust(20)
                + str(int(result[key][3])).ljust(20)
                + str(int(result[key][4])).ljust(20)
            )
