

    
import random


def calculate_0_1_label_ratio(dataset_list):
    """
    Calculates the ratio of 0s to 1s in the dataset
    """
    count0 = 0
    count1 = 0
    for tuple in dataset_list:
        if tuple[-1] == 0:
            count0 += 1
        else:
            count1 += 1
    if count0 == 0 or count1 == 0:
        print(f"Warning: count0 is {count0} and count1 is {count1}")
    return count0/count1 if count1 > 0 else count0


def oversample_minority_class(dataset_list):
    """
    Oversamples the minority class in the dataset list
    """
    majority_minority_label_ratio = calculate_0_1_label_ratio(dataset_list)
    print("Ratio of 0s to 1s in the training set: ", majority_minority_label_ratio)

    if majority_minority_label_ratio > 1:
        majority_label = 0
    else:
        majority_minority_label_ratio = 1/majority_minority_label_ratio
        majority_label = 1

    # Oversample the minority class in the dataset list
    dataset_list_copy = list(dataset_list)
    for data_tuple in dataset_list_copy:
        if data_tuple[-1] != majority_label:
            for i in range(int(majority_minority_label_ratio)):
                dataset_list.append(data_tuple)

    random.shuffle(dataset_list)
    return dataset_list

def undersample_majority_class(dataset_list):
    """
    Undersamples the majority class in the dataset list
    """
    majority_minority_label_ratio = calculate_0_1_label_ratio(dataset_list)
    print("Ratio of 0s to 1s in the training set: ", majority_minority_label_ratio)

    if majority_minority_label_ratio > 1:
        majority_label = 0
    else:
        majority_minority_label_ratio = 1/majority_minority_label_ratio
        majority_label = 1

    # Undersample the majority class in the dataset list
    dataset_list_copy = []
    for data_tuple in dataset_list:
        if data_tuple[-1] == majority_label:
            if random.randint(1, int(majority_minority_label_ratio)) == 1:
                dataset_list_copy.append(data_tuple)
        else:
            dataset_list_copy.append(data_tuple)
    dataset_list = dataset_list_copy
    random.shuffle(dataset_list)
    return dataset_list