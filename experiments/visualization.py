import matplotlib.pyplot as plt
import os


def plot_losses(net_list, net_index, losses_list, label_list, title, filename, weighted_loss, 
                images_percentage, attributes_percentage, oversample_training, 
                undersample_training, oversample_validation, undersample_validation, 
                use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                attribute_list, double_input):
    """
    Plots the losses over epochs
    """
    if preprocessing:
        type_string = "preprocessing"
    else:
        type_string = "original"

    double_nn_string = ""

    if negative_concepts:
        type_string += "_negative"

        if len(net_list) > 1:
            type_string += "_double_nn"

            if net_index % 2 == 0:
                double_nn_string = "_positive"
            else:
                double_nn_string = "_negative"

        if double_input:
            type_string += "_double_input"

    if split_attributes:
        split_attributes_string = "split_on_attributes"
    else:
        split_attributes_string = "no_split_on_attributes"

    if use_processed_concepts:
        processed_concepts_string = "processed_"
    else:
        processed_concepts_string = "original_"

    weight_sampling_string = ""
    if weighted_loss:
        weight_sampling_string += "weighted_loss_"
    if oversample_training:
        weight_sampling_string += "over_train_"
    if undersample_training:
        weight_sampling_string += "under_train_"
    if oversample_validation:
        weight_sampling_string += "over_val_"
    if undersample_validation:
        weight_sampling_string += "under_val_"

    if weight_sampling_string == "":
        weight_sampling_string = "nothing_"

    if attribute_list is None:
        percentage_string = f"{images_percentage}_{attributes_percentage}{double_nn_string}"
    else:
        percentage_string = f"custom_attributes{double_nn_string}"


    dir_path = f"plots_{type_string}/{split_attributes_string}/{processed_concepts_string}concepts/{weight_sampling_string}{percentage_string}"
    os.makedirs(dir_path, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    for i, losses in enumerate(losses_list):
        plt.plot(losses, label=label_list[i])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(dir_path, filename))
    plt.close()

def plot_metric(net_list, net_index, thresholds, metric_dict, metric_name, label, title, filename, 
                weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                undersample_training, oversample_validation, undersample_validation,
                use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                attribute_list, double_input):
    """
    Plots the metric over epochs
    """
    if preprocessing:
        type_string = "preprocessing"
    else:
        type_string = "original"

    double_nn_string = ""

    if negative_concepts:
        type_string += "_negative"

        if len(net_list) > 1:
            type_string += "_double_nn"

            if net_index % 2 == 0:
                double_nn_string = "_positive"
            else:
                double_nn_string = "_negative"

        if double_input:
            type_string += "_double_input"

    if split_attributes:
        split_attributes_string = "split_on_attributes"
    else:
        split_attributes_string = "no_split_on_attributes"

    if use_processed_concepts:
        processed_concepts_string = "processed_"
    else:
        processed_concepts_string = "original_"

    weight_sampling_string = ""
    if weighted_loss:
        weight_sampling_string += "weighted_loss_"
    if oversample_training:
        weight_sampling_string += "over_train_"
    if undersample_training:
        weight_sampling_string += "under_train_"
    if oversample_validation:
        weight_sampling_string += "over_val_"
    if undersample_validation:
        weight_sampling_string += "under_val_"

    if weight_sampling_string == "":
        weight_sampling_string = "nothing_"

    if attribute_list is None:
        percentage_string = f"{images_percentage}_{attributes_percentage}{double_nn_string}"
    else:
        percentage_string = f"custom_attributes{double_nn_string}"

    dir_path = f"plots_{type_string}/{split_attributes_string}/{processed_concepts_string}concepts/{weight_sampling_string}{percentage_string}"
    os.makedirs(dir_path, exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    for threshold in thresholds:
        plt.plot(metric_dict[threshold], label=f'{label} (Threshold {threshold})')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.title(title)
    plt.savefig(os.path.join(dir_path, filename))
    plt.close()

def plot_metrics(net_list, net_index, thresholds, metric_dicts, metric_names, labels, titles, filenames,
                weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                undersample_training, oversample_validation, undersample_validation,
                use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                attribute_list, double_input):
    """
    Plots the metrics over epochs
    """
    for i in range(len(metric_dicts)):
        plot_metric(net_list, net_index, thresholds, metric_dicts[i], metric_names[i], labels[i], titles[i], filenames[i],
                    weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                    undersample_training, oversample_validation, undersample_validation,
                    use_processed_concepts, preprocessing, negative_concepts, split_attributes, 
                    attribute_list, double_input)
        
def plot_double_nn_metrics(double_nn_scores_evolution_dict, weighted_loss, images_percentage, attributes_percentage, 
                           oversample_training, undersample_training, oversample_validation, undersample_validation, 
                           use_processed_concepts, preprocessing, split_attributes, thresholds, attribute_list):
    """
    Plots the metrics for the double NN model
    """

    double_nn_metrics_dict = {type_key: {metric_key: {threshold_key: [] for threshold_key in thresholds} for metric_key in double_nn_scores_evolution_dict[type_key][0][thresholds[0]].keys()} for type_key in double_nn_scores_evolution_dict}

    for type_key in double_nn_scores_evolution_dict:  # keys are training, validation, test
        for epoch_key in double_nn_scores_evolution_dict[type_key]:  # keys are epochs
            for threshold_key in double_nn_scores_evolution_dict[type_key][epoch_key]:  # keys are thresholds
                for metric_key in double_nn_scores_evolution_dict[type_key][epoch_key][threshold_key]:  # keys are metrics
                    metric = double_nn_scores_evolution_dict[type_key][epoch_key][threshold_key][metric_key]
                    double_nn_metrics_dict[type_key][metric_key][threshold_key].append(metric)

                    if preprocessing:
                        type_string = "preprocessing"
                    else:
                        type_string = "original"

                    type_string += "_negative_double_nn"

                    if split_attributes:
                        split_attributes_string = "split_on_attributes"
                    else:
                        split_attributes_string = "no_split_on_attributes"

                    if use_processed_concepts:
                        processed_concepts_string = "processed_"
                    else:
                        processed_concepts_string = "original_"

                    weight_sampling_string = ""
                    if weighted_loss:
                        weight_sampling_string += "weighted_loss_"
                    if oversample_training:
                        weight_sampling_string += "over_train_"
                    if undersample_training:
                        weight_sampling_string += "under_train_"
                    if oversample_validation:
                        weight_sampling_string += "over_val_"
                    if undersample_validation:
                        weight_sampling_string += "under_val_"

                    if weight_sampling_string == "":
                        weight_sampling_string = "nothing_"

                    if attribute_list is None:
                        percentage_string = f"{images_percentage}_{attributes_percentage}"
                    else:
                        percentage_string = "custom_attributes"

                    dir_path = f"plots_{type_string}/{split_attributes_string}/{processed_concepts_string}concepts/{weight_sampling_string}{percentage_string}"
                    os.makedirs(dir_path, exist_ok=True)

    for type_key in double_nn_metrics_dict:
        for metric_key in double_nn_metrics_dict[type_key]:
            plt.figure(figsize=(10, 5))
            for threshold_key in thresholds:
                plt.plot(double_nn_metrics_dict[type_key][metric_key][threshold_key], label=f'Threshold {threshold_key}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_key.capitalize())
            plt.legend()
            plt.title(f'{type_key.capitalize()} {metric_key.capitalize()} Over Epochs')
            plt.savefig(os.path.join(dir_path, f'{type_key}_{metric_key}.png'))
            plt.close()
