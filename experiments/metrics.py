import torch
from sklearn.metrics import f1_score


def f1_score_manual(labels, predictions):
    """
    Calculates the F1 score
    """
    labels_tensor = torch.tensor(labels)
    predictions_tensor = torch.tensor(predictions)
    
    label_1_prediction_1 = torch.sum((labels_tensor == 1) & (predictions_tensor == 1)).item()
    label_1_prediction_0 = torch.sum((labels_tensor == 1) & (predictions_tensor == 0)).item()
    label_0_prediction_1 = torch.sum((labels_tensor == 0) & (predictions_tensor == 1)).item()

    
    if label_1_prediction_1 + label_0_prediction_1 == 0:
        precision = 0
    else:
        precision = label_1_prediction_1 / (label_1_prediction_1 + label_0_prediction_1)

    if label_1_prediction_1 + label_1_prediction_0 == 0:
        recall = 0
    else:
        recall = label_1_prediction_1 / (label_1_prediction_1 + label_1_prediction_0)

    if precision + recall == 0:
        f1 =  0
    else: 
        f1 = 2 * (precision * recall) / (precision + recall)

    sklearn_f1 = f1_score(labels, predictions)
    if round(f1, 4) != round(sklearn_f1, 4):
        print("Warning: Manual F1 score and Sklearn F1 score are not equal")
        print("Manual F1 score: ", round(f1, 4))
        print("Sklearn F1 score: ", round(sklearn_f1, 4))

    return round(sklearn_f1, 4)

def calculate_double_nn_scores(double_nn_output_dict, all_original_labels=None, thresholds=None):
    """
    Calculates the scores for the double NN model
    """
    if thresholds is not None:
        double_nn_scores_dict = {}
        prediction_dict = {}
        for threshold in thresholds:
            double_nn_scores_dict[threshold] = {}
            prediction_dict[threshold] = []
    else:
        double_nn_scores_dict = None
        prediction_dict = None

    average_output = [0] * len(double_nn_output_dict[0])
    for net_index in range(len(double_nn_output_dict)):
        if net_index % 2 == 0:
            average_output = [average_output[i] + double_nn_output_dict[net_index][i]/len(double_nn_output_dict) for i in range(len(average_output))]
        else:
            average_output = [average_output[i] + (1 - double_nn_output_dict[net_index][i])/len(double_nn_output_dict) for i in range(len(average_output))]

    if thresholds is not None:
        for threshold in thresholds:
            threshold_output = torch.where(torch.tensor(average_output) > threshold, torch.tensor(1.0), torch.tensor(0.0))
            threshold_output = threshold_output.squeeze()
            prediction_dict[threshold] = threshold_output.cpu().numpy()
            if all_original_labels is not None:
                label_1_prediction_1 = 0
                label_1_prediction_0 = 0
                label_0_prediction_1 = 0
                label_0_prediction_0 = 0
                for label, prediction in zip(all_original_labels, threshold_output):
                    if label == 1 and prediction == 1:
                        label_1_prediction_1 += 1
                    elif label == 1 and prediction == 0:
                        label_1_prediction_0 += 1
                    elif label == 0 and prediction == 1:
                        label_0_prediction_1 += 1
                    elif label == 0 and prediction == 0:
                        label_0_prediction_0 += 1

                accuracy = (label_1_prediction_1 + label_0_prediction_0) / (
                    label_1_prediction_1 + label_1_prediction_0 + label_0_prediction_1 + label_0_prediction_0
                )
                accuracy_0 = label_0_prediction_0 / (label_0_prediction_0 + label_0_prediction_1) if (label_0_prediction_0 + label_0_prediction_1) > 0 else 0
                accuracy_1 = label_1_prediction_1 / (label_1_prediction_1 + label_1_prediction_0) if (label_1_prediction_1 + label_1_prediction_0) > 0 else 0
                balanced_accuracy = (accuracy_0 + accuracy_1) / 2
                f1 = f1_score_manual(all_original_labels, threshold_output.cpu().numpy())

                double_nn_scores_dict[threshold]['accuracy'] = accuracy
                double_nn_scores_dict[threshold]['accuracy_0'] = accuracy_0
                double_nn_scores_dict[threshold]['accuracy_1'] = accuracy_1
                double_nn_scores_dict[threshold]['balanced_accuracy'] = balanced_accuracy
                double_nn_scores_dict[threshold]['f1'] = f1

    return average_output, prediction_dict, double_nn_scores_dict
