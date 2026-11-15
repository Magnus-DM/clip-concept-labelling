
import torch
import torch.nn as nn


class AttributeClassifier(nn.Module):
    def __init__(self, preprocessing, input_size, regularization, dropout_rate):
        super(AttributeClassifier, self).__init__()
        if not preprocessing:
            self.fc1 = nn.Linear(input_size, 512)
            if regularization:
                self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(512, 256)
            if regularization:
                self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(256, 1)
            self.activation = nn.Sigmoid()
        elif preprocessing:
            self.fc1 = nn.Linear(input_size, 10)
            if regularization:
                self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(10, 10)
            if regularization:
                self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(10, 1)
            self.activation = nn.Sigmoid()

    def forward(self, input_list, regularization):
        if len(input_list) == 1:
            x = torch.relu(self.fc1(input_list[0]))
            
            x = torch.relu(self.fc2(x))
            
            x = self.activation(self.fc3(x))
            return x
        else:
            x = input_list[0]
            for i in range(1, len(input_list)):
                x = torch.cat((x, input_list[i]), dim=2)
            x = torch.relu(self.fc1(x))
            
            x = torch.relu(self.fc2(x))
            
            x = self.activation(self.fc3(x))
            return x

    # def forward(self, input_list):
    #     if len(input_list) == 1:
    #         x = torch.relu(self.fc1(input_list[0]))
    #         x = torch.relu(self.fc2(x))
    #         x = self.activation(self.fc3(x))
    #         return x
    #     else:
    #         x = input_list[0]
    #         for i in range(1, len(input_list)):
    #             x = torch.cat((x, input_list[i]), dim=2)
    #         x = torch.relu(self.fc1(x))
    #         x = torch.relu(self.fc2(x))
    #         x = self.activation(self.fc3(x))
    #         return x