import torch
import torch.nn as nn
from metrics import f1_score_manual, calculate_double_nn_scores
from visualization import plot_metrics, plot_double_nn_metrics, plot_losses

def test_model(device, net_list, net_index, test_loader, weighted_loss=False, majority_label=0, set_maj_min_label_ratio=1, 
               thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], preprocessing=False, double_nn_output_dict=None, double_input=False,
               regularization=False):
    """
    Tests the model on the test data (or validation data)
    """
    all_original_labels = [] if len(net_list) > 1 else None
    net_list[net_index].eval()
    total_loss = 0
    total_unweighted_loss = 0
    total_items = 0
    all_labels = []
    all_predictions = {}
    total_differences = {}
    total_label_1_prediction_1 = {}
    total_label_1_prediction_0 = {}
    total_label_0_prediction_1 = {}
    total_label_0_prediction_0 = {}
    accuracy_0 = {}
    accuracy_1 = {}
    accuracy = {}
    f1 = {}
    for threshold in thresholds:
        all_predictions[threshold] = []
        total_differences[threshold] = 0
        total_label_1_prediction_1[threshold] = 0
        total_label_1_prediction_0[threshold] = 0
        total_label_0_prediction_1[threshold] = 0
        total_label_0_prediction_0[threshold] = 0

    with torch.no_grad():
        for batch_idx, data_tuple in enumerate(test_loader):
            input_list = list(data_tuple[:-1])
            if len(net_list) > 1:
                if not preprocessing:
                    input_list = [input_list[0]] + [input_list[i] for i in range(1, len(input_list)) if (i + net_index) % 2 != 0]
                else:
                    input_tensor = input_list[0]
                    input_tensor = input_tensor[:, [(i + net_index) % 2 == 0 for i in range(input_tensor.shape[1])]]
                    input_list = [input_tensor]
            input_list = [tensor.to(device) for tensor in input_list]
            label = data_tuple[-1]

            if len(net_list) > 1:
                all_original_labels.extend(label.cpu().numpy())
            if net_index % 2 != 0:
                label = 1 - label
            input_list = [tensor.to(device) for tensor in input_list]
            label = label.to(device)
            label = label.float().squeeze()
            output = net_list[net_index](input_list, regularization)
            output = output.squeeze()
            output = output.to(device)

            if double_input:
                if isinstance(output, torch.Tensor):
                    output_joint = (output[::2] + (1 - output[1::2])) / 2
                    output_joint = output_joint.squeeze()
                    output_joint = output_joint.to(device)
                else:
                    raise ValueError("output is not a tensor")

            if len(net_list) > 1:
                double_nn_output_dict[net_index].extend(output.detach().cpu().numpy())
            threshold_output = {}
            for threshold in thresholds:
                if double_input:
                    threshold_output[threshold] = torch.where(output_joint > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                    threshold_output[threshold] = threshold_output[threshold].squeeze()
                else:
                    threshold_output[threshold] = torch.where(output > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                    threshold_output[threshold] = threshold_output[threshold].squeeze()
            
            if weighted_loss:
                weight = torch.where(label == majority_label, torch.tensor(1.0, device=device), torch.tensor(set_maj_min_label_ratio, device=device))
                loss = nn.BCELoss(weight=weight)(output, label)
            else:
                loss = nn.BCELoss()(output, label)
            unweighted_loss = nn.BCELoss()(output, label)
            total_loss += loss.item()
            total_unweighted_loss += unweighted_loss.item()

            if double_input:
                if isinstance(label, torch.Tensor):
                    label = label[::2]
                    label = label.float().squeeze()
                    label = label.to(device)
                else:
                    raise ValueError("label is not a tensor")
            
            # Calculate the number of differences
            differences = {}
            for threshold in thresholds:
                differences[threshold] = torch.sum(threshold_output[threshold] != label)
                total_differences[threshold] += differences[threshold].item()

            # Calculate TN, FP, FN, TP for each threshold:
            label_1_prediction_1 = {}
            label_1_prediction_0 = {}
            label_0_prediction_1 = {}
            label_0_prediction_0 = {}
            for threshold in thresholds:
                label_1_prediction_1[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 1))
                label_1_prediction_0[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 0))
                label_0_prediction_1[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 1))
                label_0_prediction_0[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 0))
                total_label_1_prediction_1[threshold] += label_1_prediction_1[threshold].item()
                total_label_1_prediction_0[threshold] += label_1_prediction_0[threshold].item()
                total_label_0_prediction_1[threshold] += label_0_prediction_1[threshold].item()
                total_label_0_prediction_0[threshold] += label_0_prediction_0[threshold].item()

            total_items += label.numel()
            
            # Collect all labels and predictions for F1 score calculation
            all_labels.extend(label.cpu().numpy())

            for threshold in thresholds:
                all_predictions[threshold].extend(threshold_output[threshold].cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    avg_unweighted_loss = total_unweighted_loss / len(test_loader)

    for threshold in thresholds:
        accuracy[threshold] = 1 - (total_differences[threshold] / total_items)
        accuracy_0[threshold] = total_label_0_prediction_0[threshold] / (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) \
            if (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) > 0 else 0
        accuracy_1[threshold] = total_label_1_prediction_1[threshold] / (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) \
            if (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) > 0 else 0
        f1[threshold] = f1_score_manual(all_labels, all_predictions[threshold])
    
    return avg_loss, avg_unweighted_loss, accuracy, accuracy_0, accuracy_1, f1, double_nn_output_dict, all_original_labels

def train_model(device, net_list, optimizer_list, preprocessing, negative_concepts, train_loader, validation_loader, 
                test_loader, epochs=30, weighted_loss=False, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0], training_majority_label=0, training_set_maj_min_label_ratio=1,
                validation_majority_label=0, validation_set_maj_min_label_ratio=1, test_majority_label=0, test_set_maj_min_label_ratio=1, 
                images_percentage=100, attributes_percentage=100, oversample_training=False, undersample_training=False, 
                oversample_validation=False, undersample_validation=False, use_processed_concepts=True, split_attributes=False, attribute_list=None,
                double_input=False, regularization=False):
    """
    Trains the model on the training data and validates it on the validation data
    """
    original_training_majority_label = training_majority_label
    original_validation_majority_label = validation_majority_label
    original_test_majority_label = test_majority_label

    double_nn_training_output_dict_epoch_list = [None] * epochs
    double_nn_validation_output_dict_epoch_list = [None] * epochs
    double_nn_test_output_dict_epoch_list = [None] * epochs
    if len(net_list) > 1:
        double_nn_training_output_dict_epoch_list = [] # Honestly, even I don't know what the fuck I'm doing here
        double_nn_validation_output_dict_epoch_list = []
        double_nn_test_output_dict_epoch_list = []
        for epoch in range(epochs):
            double_nn_training_output_dict_epoch_list.append({})
            double_nn_validation_output_dict_epoch_list.append({})
            double_nn_test_output_dict_epoch_list.append({})
            for net_index in range(len(net_list)):
                double_nn_training_output_dict_epoch_list[epoch][net_index] = []
                double_nn_validation_output_dict_epoch_list[epoch][net_index] = []
                double_nn_test_output_dict_epoch_list[epoch][net_index] = []
        double_nn_scores_evolution_dict = {}
        double_nn_scores_evolution_dict["training"] = {}
        double_nn_scores_evolution_dict["validation"] = {}
        double_nn_scores_evolution_dict["test"] = {}
        best_nets = []
        all_original_training_labels = []
        original_labels_complete = False

    for net_index in range(len(net_list)):
        print(f"Training network {net_index + 1} of {len(net_list)}")
        print("--------------------")
        net_list[net_index].train()

        best_validation_loss = float('inf')
        best_model_state = None

        # Lists to store losses and metrics
        training_losses = [] 
        training_unweighted_losses = [] 
        validation_losses = [] 
        validation_unweighted_losses = [] 
        test_losses = [] 
        test_unweighted_losses = [] 
        training_accuracies = {}
        validation_accuracies = {}
        test_accuracies = {}
        training_accuracies_0 = {}
        validation_accuracies_0 = {}
        test_accuracies_0 = {}
        training_accuracies_1 = {}
        validation_accuracies_1 = {}
        test_accuracies_1 = {}
        training_balanced_accuracies = {}
        validation_balanced_accuracies = {}
        test_balanced_accuracies = {}
        training_f1_scores = {}
        validation_f1_scores = {}
        test_f1_scores = {}
        for threshold in thresholds:
            training_accuracies[threshold] = [] 
            validation_accuracies[threshold] = [] 
            test_accuracies[threshold] = [] 
            training_accuracies_0[threshold] = [] 
            validation_accuracies_0[threshold] = [] 
            test_accuracies_0[threshold] = [] 
            training_accuracies_1[threshold] = [] 
            validation_accuracies_1[threshold] = [] 
            test_accuracies_1[threshold] = [] 
            training_balanced_accuracies[threshold] = [] 
            validation_balanced_accuracies[threshold] = [] 
            test_balanced_accuracies[threshold] = [] 
            training_f1_scores[threshold] = [] 
            validation_f1_scores[threshold] = [] 
            test_f1_scores[threshold] = [] 

        for epoch in range(epochs):
            total_loss = 0 
            total_unweighted_loss = 0 
            total_items = 0
            all_labels = []
            total_differences = {}
            all_predictions = {}
            total_label_1_prediction_1 = {}
            total_label_1_prediction_0 = {}
            total_label_0_prediction_1 = {}
            total_label_0_prediction_0 = {}
            for threshold in thresholds:
                all_predictions[threshold] = [] 
                total_differences[threshold] = 0 
                total_label_1_prediction_1[threshold] = 0 
                total_label_1_prediction_0[threshold] = 0 
                total_label_0_prediction_1[threshold] = 0 
                total_label_0_prediction_0[threshold] = 0 

            print(f"Starting epoch {epoch+1}/{epochs}")
            for batch_idx, data_tuple in enumerate(train_loader):
                input_list = list(data_tuple[:-1])
                if len(net_list) > 1:
                    if not preprocessing:
                        input_list = [input_list[0]] + [input_list[i] for i in range(1, len(input_list)) if (i + net_index) % 2 != 0]
                    else:
                        input_tensor = input_list[0]
                        input_tensor = input_tensor[:, [(i + net_index) % 2 == 0 for i in range(input_tensor.shape[1])]]
                        input_list = [input_tensor]
                input_list = [tensor.to(device) for tensor in input_list]
                label = data_tuple[-1]
                if net_index % 2 != 0:
                    label = 1 - label
                    training_majority_label = 1 - original_training_majority_label
                    validation_majority_label = 1 - original_validation_majority_label
                    test_majority_label = 1 - original_test_majority_label
                label = label.to(device)
                label = label.float().squeeze()
                optimizer_list[net_index].zero_grad()
                output = net_list[net_index](input_list, regularization)
                output = output.squeeze()
                output = output.to(device)
                
                if len(net_list) > 1:
                    double_nn_training_output_dict_epoch_list[epoch][net_index].extend(output.detach().cpu().numpy())
                    if net_index == 0 and not original_labels_complete:
                        all_original_training_labels.extend(label.cpu().numpy())

                if weighted_loss:
                    weight = torch.where(label == training_majority_label, torch.tensor(1.0, device=device), torch.tensor(training_set_maj_min_label_ratio, device=device))
                    loss = nn.BCELoss(weight=weight)(output, label)
                else:
                    loss = nn.BCELoss()(output, label)
                unweighted_loss = nn.BCELoss()(output, label)
                loss.backward()
                optimizer_list[net_index].step()
                total_loss += loss.item()
                total_unweighted_loss += unweighted_loss.item()

                if double_input:
                    if isinstance(output, torch.Tensor):
                        output_joint = (output[::2] + (1 - output[1::2])) / 2
                        output_joint = output_joint.squeeze()
                        output_joint = output_joint.to(device)
                    else:
                        raise ValueError("output is not a tensor")
                    
                    if isinstance(label, torch.Tensor):
                        label_joint = label[::2]
                        label_joint = label_joint.float().squeeze()
                        label_joint = label_joint.to(device)
                    else:
                        raise ValueError("label is not a tensor")

                # Calculate the number of differences
                threshold_output = {}
                differences = {}
                for threshold in thresholds:
                    if double_input:
                        threshold_output[threshold] = torch.where(output_joint > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                        threshold_output[threshold] = threshold_output[threshold].squeeze()
                        differences[threshold] = torch.sum(threshold_output[threshold] != label_joint)
                    else:
                        threshold_output[threshold] = torch.where(output > threshold, torch.tensor(1.0, device=output.device), torch.tensor(0.0, device=output.device))
                        threshold_output[threshold] = threshold_output[threshold].squeeze()
                        differences[threshold] = torch.sum(threshold_output[threshold] != label)
                    total_differences[threshold] += differences[threshold].item()

                # Calculate TN, FP, FN, TP for each threshold:
                label_1_prediction_1 = {}
                label_1_prediction_0 = {}
                label_0_prediction_1 = {}
                label_0_prediction_0 = {}
                for threshold in thresholds:
                    if double_input:
                        label_1_prediction_1[threshold] = torch.sum((label_joint == 1) & (threshold_output[threshold] == 1))
                        label_1_prediction_0[threshold] = torch.sum((label_joint == 1) & (threshold_output[threshold] == 0))
                        label_0_prediction_1[threshold] = torch.sum((label_joint == 0) & (threshold_output[threshold] == 1))
                        label_0_prediction_0[threshold] = torch.sum((label_joint == 0) & (threshold_output[threshold] == 0))
                    else:
                        label_1_prediction_1[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 1))
                        label_1_prediction_0[threshold] = torch.sum((label == 1) & (threshold_output[threshold] == 0))
                        label_0_prediction_1[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 1))
                        label_0_prediction_0[threshold] = torch.sum((label == 0) & (threshold_output[threshold] == 0))
                    total_label_1_prediction_1[threshold] += label_1_prediction_1[threshold].item()
                    total_label_1_prediction_0[threshold] += label_1_prediction_0[threshold].item()
                    total_label_0_prediction_1[threshold] += label_0_prediction_1[threshold].item()
                    total_label_0_prediction_0[threshold] += label_0_prediction_0[threshold].item()

                if double_input:
                    total_items += label_joint.numel()
                else:
                    total_items += label.numel()
                
                # Collect all labels and predictions for F1 score calculation
                if double_input:
                    all_labels.extend(label_joint.cpu().numpy())
                else:
                    all_labels.extend(label.cpu().numpy())

                for threshold in thresholds:
                    all_predictions[threshold].extend(threshold_output[threshold].cpu().numpy())
                
                # if batch_idx % 10 == 0:  # Print every 10 batches
                #     print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Batch Loss: {loss.item()}")
            
            if len(net_list) > 1 and all_original_training_labels != []:
                original_labels_complete = True

            avg_loss = total_loss / len(train_loader)
            avg_unweighted_loss = total_unweighted_loss / len(train_loader)
            training_losses.append(avg_loss)
            training_unweighted_losses.append(avg_unweighted_loss)
            # print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
            # print(f"Epoch {epoch+1} completed, Average Unweighted Loss: {avg_unweighted_loss:.4f}")
            
            # Calculate training accuracy and F1 score
            training_accuracy = {}
            training_accuracy_0 = {}
            training_accuracy_1 = {}
            training_f1_score = {}
            for threshold in thresholds:
                training_accuracy[threshold] = 1 - (total_differences[threshold] / total_items)
                training_accuracy_0[threshold] = total_label_0_prediction_0[threshold] / (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) \
                    if (total_label_0_prediction_0[threshold] + total_label_0_prediction_1[threshold]) > 0 else 0
                training_accuracy_1[threshold] = total_label_1_prediction_1[threshold] / (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) \
                    if (total_label_1_prediction_1[threshold] + total_label_1_prediction_0[threshold]) > 0 else 0
                training_f1_score[threshold] = f1_score_manual(all_labels, all_predictions[threshold])
                training_accuracies[threshold].append(training_accuracy[threshold])
                training_accuracies_0[threshold].append(training_accuracy_0[threshold])
                training_accuracies_1[threshold].append(training_accuracy_1[threshold])
                training_balanced_accuracies[threshold].append((training_accuracy_0[threshold] + training_accuracy_1[threshold]) / 2)
                training_f1_scores[threshold].append(training_f1_score[threshold])

            # Validate the model on the validation data
            validation_loss, validation_unweighted_loss, validation_accuracy, validation_accuracy_0, validation_accuracy_1, validation_f1, \
            double_nn_validation_output_dict, all_original_validation_labels = test_model(device,net_list, net_index, validation_loader, 
                weighted_loss, validation_majority_label, validation_set_maj_min_label_ratio, thresholds, preprocessing, 
                double_nn_validation_output_dict_epoch_list[epoch], double_input, regularization)
            
            if len(net_list) > 1:
                double_nn_validation_output_dict_epoch_list[epoch] = double_nn_validation_output_dict

            validation_losses.append(validation_loss)
            validation_unweighted_losses.append(validation_unweighted_loss)
            for threshold in thresholds:
                validation_accuracies[threshold].append(validation_accuracy[threshold])
                validation_accuracies_0[threshold].append(validation_accuracy_0[threshold])
                validation_accuracies_1[threshold].append(validation_accuracy_1[threshold])
                validation_balanced_accuracies[threshold].append((validation_accuracy_0[threshold] + validation_accuracy_1[threshold]) / 2)
                validation_f1_scores[threshold].append(validation_f1[threshold])

            # Test the model on the test data
            test_loss, test_unweighted_loss, test_accuracy, test_accuracy_0, test_accuracy_1, test_f1, \
            double_nn_test_output_dict, all_original_test_labels = test_model(device,net_list, net_index, 
                test_loader, weighted_loss, test_majority_label, test_set_maj_min_label_ratio, thresholds, preprocessing, 
                double_nn_test_output_dict_epoch_list[epoch], double_input, regularization)
            
            if len(net_list) > 1:
                double_nn_test_output_dict_epoch_list[epoch] = double_nn_test_output_dict

            test_losses.append(test_loss)
            test_unweighted_losses.append(test_unweighted_loss)
            for threshold in thresholds:
                test_accuracies[threshold].append(test_accuracy[threshold])
                test_accuracies_0[threshold].append(test_accuracy_0[threshold])
                test_accuracies_1[threshold].append(test_accuracy_1[threshold])
                test_balanced_accuracies[threshold].append((test_accuracy_0[threshold] + test_accuracy_1[threshold]) / 2)
                test_f1_scores[threshold].append(test_f1[threshold])

            if len(net_list) > 1 and net_index == len(net_list) - 1:
                average_training_output, training_prediction_dict, double_nn_scores_evolution_dict["training"][epoch] = \
                calculate_double_nn_scores(double_nn_training_output_dict_epoch_list[epoch], all_original_training_labels, thresholds)
                average_validation_output, validation_prediction_dict, double_nn_scores_evolution_dict["validation"][epoch] = \
                calculate_double_nn_scores(double_nn_validation_output_dict_epoch_list[epoch], all_original_validation_labels, thresholds)
                average_test_output, test_prediction_dict, double_nn_scores_evolution_dict["test"][epoch] = \
                calculate_double_nn_scores(double_nn_test_output_dict_epoch_list[epoch], all_original_test_labels, thresholds)

            # Save the model if it has the best validation loss so far
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_model_state = net_list[net_index].state_dict()

            # print("Training Losses: ", training_losses)
            # print("Validation Losses: ", validation_losses)
            # print("Test Losses: ", test_losses)
            # print("Training Unweighted Losses: ", training_unweighted_losses)
            # print("Validation Unweighted Losses: ", validation_unweighted_losses)
            # print("Test Unweighted Losses: ", test_unweighted_losses)
            # for threshold in thresholds:
            #     print(f"Training Accuracies with threshold {threshold}: ", training_accuracies[threshold])
            #     print(f"Validation Accuracies with threshold {threshold}: ", validation_accuracies[threshold])
            #     print(f"Test Accuracies with threshold {threshold}: ", test_accuracies[threshold])
            #     print(f"Training Accuracies (0) with threshold {threshold}: ", training_accuracies_0[threshold])
            #     print(f"Validation Accuracies (0) with threshold {threshold}: ", validation_accuracies_0[threshold])
            #     print(f"Test Accuracies (0) with threshold {threshold}: ", test_accuracies_0[threshold])
            #     print(f"Training Accuracies (1) with threshold {threshold}: ", training_accuracies_1[threshold])
            #     print(f"Validation Accuracies (1) with threshold {threshold}: ", validation_accuracies_1[threshold])
            #     print(f"Test Accuracies (1) with threshold {threshold}: ", test_accuracies_1[threshold])
            #     print(f"Training Balanced Accuracies with threshold {threshold}: ", training_balanced_accuracies[threshold])
            #     print(f"Validation Balanced Accuracies with threshold {threshold}: ", validation_balanced_accuracies[threshold])
            #     print(f"Test Balanced Accuracies with threshold {threshold}: ", test_balanced_accuracies[threshold])
            #     print(f"Training F1 Scores with threshold {threshold}: ", training_f1_scores[threshold])
            #     print(f"Validation F1 Scores with threshold {threshold}: ", validation_f1_scores[threshold])
            #     print(f"Test F1 Scores with threshold {threshold}: ", test_f1_scores[threshold])

        # Save the losses plot
        losses_list = [training_losses, validation_losses, test_losses, training_unweighted_losses, validation_unweighted_losses, test_unweighted_losses]
        label_list = ['Training Loss', 'Validation Loss', 'Test Loss', 'Training Unweighted Loss', 'Validation Unweighted Loss', 'Test Unweighted Loss']
        title = 'Training, Validation, and Test Loss Over Epochs'
        filename = 'losses.png'
        plot_losses(net_list, net_index, losses_list, label_list, title, filename, weighted_loss, 
                    images_percentage, attributes_percentage, oversample_training, 
                    undersample_training, oversample_validation, undersample_validation, 
                    use_processed_concepts, preprocessing, negative_concepts, split_attributes, attribute_list,
                    double_input)

        # Save the accuracies and F1 scores plots
        metric_dicts = [training_accuracies, validation_accuracies, test_accuracies,
                        training_accuracies_0, validation_accuracies_0, test_accuracies_0,
                        training_accuracies_1, validation_accuracies_1, test_accuracies_1,
                        training_balanced_accuracies, validation_balanced_accuracies, test_balanced_accuracies,
                        training_f1_scores, validation_f1_scores, test_f1_scores]
        metric_names = ['Accuracy', 'Accuracy', 'Accuracy', 'Accuracy (0)', 'Accuracy (0)', 'Accuracy (0)', 
                        'Accuracy (1)', 'Accuracy (1)', 'Accuracy (1)', 'Balanced Accuracy', 'Balanced Accuracy', 'Balanced Accuracy', 'F1 Score', 'F1 Score', 'F1 Score']
        labels = ['Training', 'Validation', 'Test', 'Training', 'Validation', 'Test', 'Training', 'Validation', 'Test', 'Training', 'Validation', 'Test', 'Training', 'Validation', 'Test']
        titles = ['Training Accuracy Over Epochs', 'Validation Accuracy Over Epochs', 'Test Accuracy Over Epochs',
                'Training Accuracy (0) Over Epochs', 'Validation Accuracy (0) Over Epochs', 'Test Accuracy (0) Over Epochs',
                'Training Accuracy (1) Over Epochs', 'Validation Accuracy (1) Over Epochs', 'Test Accuracy (1) Over Epochs',
                'Training Balanced Accuracy Over Epochs', 'Validation Balanced Accuracy Over Epochs', 'Test Balanced Accuracy Over Epochs',
                'Training F1 Score Over Epochs', 'Validation F1 Score Over Epochs', 'Test F1 Score Over Epochs']
        filenames = ['training_accuracies.png', 'validation_accuracies.png', 'test_accuracies.png',
                    'training_accuracies_0.png', 'validation_accuracies_0.png', 'test_accuracies_0.png',
                    'training_accuracies_1.png', 'validation_accuracies_1.png', 'test_accuracies_1.png',
                    'training_balanced_accuracies.png', 'validation_balanced_accuracies.png', 'test_balanced_accuracies.png',
                    'training_f1_scores.png', 'validation_f1_scores.png', 'test_f1_scores.png']
        plot_metrics(net_list, net_index, thresholds, metric_dicts, metric_names, labels, titles, filenames, 
                    weighted_loss, images_percentage, attributes_percentage, oversample_training, 
                    undersample_training, oversample_validation, undersample_validation, 
                    use_processed_concepts, preprocessing, negative_concepts, split_attributes, attribute_list,
                    double_input)

        # Load the best model state
        if best_model_state is not None:
            net_list[net_index].load_state_dict(best_model_state)
            print("Loaded the best model with validation loss: {:.4f}".format(best_validation_loss))
            if len(net_list) > 1:
                best_nets.append(best_model_state)

        if len(net_list) == 1:
            # Test the model on the test data
            test_loss, test_unweighted_loss, test_accuracy, test_accuracy_0, test_accuracy_1, test_f1, \
            double_nn_test_output_dict, all_original_test_labels = test_model(device,net_list, net_index, 
                test_loader, weighted_loss, test_majority_label, test_set_maj_min_label_ratio, thresholds, preprocessing, 
                None, double_input, regularization)

            print(f"Test Loss on best model: {test_loss:.4f}")
            print(f"Test Unweighted Loss on best model: {test_unweighted_loss:.4f}")
            for threshold in thresholds:
                #print(f"Test Accuracy with threshold {threshold} on best model: {test_accuracy[threshold]:.4f}")
                #print(f"Test Accuracy (0) with threshold {threshold} on best model: {test_accuracy_0[threshold]:.4f}")
                #print(f"Test Accuracy (1) with threshold {threshold} on best model: {test_accuracy_1[threshold]:.4f}")
                #print(f"Test Balanced Accuracy with threshold {threshold} on best model: {(test_accuracy_0[threshold] + test_accuracy_1[threshold]) / 2:.4f}")
                #print(f"Test F1 score with threshold {threshold} on best model: {test_f1[threshold]:.4f}")
                print(f"Test accuracies for theshold {threshold}: (0) = {test_accuracy_0[threshold]:.4f}, (1) = {test_accuracy_1[threshold]:.4f}, (balanced) = {(test_accuracy_0[threshold] + test_accuracy_1[threshold]) / 2:.4f}, (F1) = {test_f1[threshold]:.4f}")
                print(f"Train accuracies for theshold {threshold}: (0) = {training_accuracy_0[threshold]:.4f}, (1) = {training_accuracy_1[threshold]:.4f}, (balanced) = {(training_accuracy_0[threshold] + training_accuracy_1[threshold]) / 2:.4f}, (F1) = {training_f1_score[threshold]:.4f}")

    if len(net_list) > 1:
        print("Plotting metrics for the double NN")
        plot_double_nn_metrics(double_nn_scores_evolution_dict, weighted_loss, images_percentage, attributes_percentage, 
                               oversample_training, undersample_training, oversample_validation, undersample_validation, 
                               use_processed_concepts, preprocessing, split_attributes, thresholds, attribute_list)
        print("Plotting metrics for the double NN completed")
    
        for net_index in range(len(net_list)):
            double_nn_test_output_dict[net_index] = []
        
        for net_index in range(len(net_list)):
            if net_index % 2 != 0:
                test_majority_label = 1 - original_test_majority_label
            net_list[net_index].load_state_dict(best_nets[net_index])
            test_loss, test_unweighted_loss, test_accuracy, test_accuracy_0, test_accuracy_1, test_f1, \
            double_nn_test_output_dict, all_original_test_labels = test_model(device,net_list, net_index,  
                test_loader, weighted_loss, test_majority_label, test_set_maj_min_label_ratio, thresholds, preprocessing, 
                double_nn_test_output_dict, double_input, regularization)

        average_test_output, test_prediction_dict, double_nn_scores_dict = \
        calculate_double_nn_scores(double_nn_test_output_dict, all_original_test_labels, thresholds)

        for threshold_key in double_nn_scores_dict:
            #print(f"Test Accuracy with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['accuracy']:.4f}")
            #print(f"Test Accuracy (0) with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['accuracy_0']:.4f}")
            #print(f"Test Accuracy (1) with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['accuracy_1']:.4f}")
            #print(f"Test Balanced Accuracy with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['balanced_accuracy']:.4f}")
            #print(f"Test F1 score with threshold {threshold_key} on best model: {double_nn_scores_dict[threshold_key]['f1']:.4f}")
            print(f"Test accuracies for theshold {threshold_key}: (0) = {double_nn_scores_dict[threshold_key]['accuracy_0']:.4f}, (1) = {double_nn_scores_dict[threshold_key]['accuracy_1']:.4f}, (balanced) = {double_nn_scores_dict[threshold_key]['balanced_accuracy']}, (F1) = {double_nn_scores_dict[threshold_key]['f1']:.4f}")
