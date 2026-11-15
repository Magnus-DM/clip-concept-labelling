import csv
import re


def cub_to_csv():
    # Input and output file paths
    input_file = './original_files/results_im_percentage_100.txt'
    output_file = './cub200_attribute_accuracies.csv'

    # Regular expression to extract the needed parts
    pattern = re.compile(
        r'Attribute (\d+)\. (.*?): Threshold = ([\d.]+), Balanced Accuracy = ([\d.]+)'
    )


    # Open the input and output files
    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['attribute_number', 'label', 'threshold', 'balanced_accuracy'])

        # Go through each line and extract data
        for line in infile:
            match = pattern.search(line)
            if match:
                attr_num = match.group(1)
                label = match.group(2)
                threshold = match.group(3)
                balanced_accuracy = match.group(4)
                writer.writerow([attr_num, label, threshold, balanced_accuracy])

    print(f'Data has been written to {output_file}')


def celebA_to_csv():
    # Input file paths
    accuracy_file = './original_files/celebA_attribute_accuracy.txt'
    attributes_file = './attributes_celebA.txt'
    output_file = './celebA_attribute_accuracies.csv'

    # Load attribute labels into a dictionary
    attr_labels = {}
    with open(attributes_file, 'r') as af:
        for line in af:
            parts = line.strip().split(maxsplit=1)
            print(parts)
            attr_num, label = parts
            attr_labels[attr_num] = label

    # Regular expression to extract data from accuracy file
    pattern = re.compile(
        r'\d+\s+(\d+)\s+([\d.]+)\s+([\d.]+)'
    )

    # Process the accuracy file and write to CSV
    with open(accuracy_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # Write header
        writer.writerow(['attribute_number', 'label', 'balanced_accuracy', 'accuracy'])

        for line in infile:
            match = pattern.match(line.strip())
            if match:
                attr_num = str(int(match.group(1)) + 1)
                balanced_acc = match.group(2)
                accuracy = match.group(3)
                label = attr_labels.get(attr_num, 'Unknown')
                print(attr_num, label, balanced_acc, accuracy)
                writer.writerow([attr_num, label, balanced_acc, accuracy])

    print(f'Data with labels has been written to {output_file}')

def celebA_labels_only():

    # Input and output file paths
    input_file = 'celebA_attribute_accuracies.csv'
    output_file = './celebA_attributes.csv'

    # Read the data
    attributes = []
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            attr_num = int(row['attribute_number'])
            label = row['label']
            attributes.append((attr_num, label))

    # Sort by attribute number
    attributes.sort(key=lambda x: x[0])

    # Write to new CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['attribute_number', 'label'])
        for attr_num, label in attributes:
            writer.writerow([attr_num, label])

    print(f'Labels-only CSV written to {output_file}')


def cub_labels_only():
    # Input and output file paths
    input_file = './cub200_attribute_accuracies.csv'
    output_file = './cub200_attributes.csv'

    # Read the data
    attributes = []
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            attr_num = int(row['attribute_number'])
            label = row['label']
            attributes.append((attr_num, label))

    # Sort by attribute number
    attributes.sort(key=lambda x: x[0])

    # Write to new CSV
    with open(output_file, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['attribute_number', 'label'])
        for attr_num, label in attributes:
            writer.writerow([attr_num, label])

    print(f'Labels-only CSV written to {output_file}')

if __name__ == "__main__":
    # Uncomment the function you want to run
    cub_to_csv()
    celebA_to_csv()
    celebA_labels_only()
    cub_labels_only()