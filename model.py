import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, input, lamb):
        ctx.lamb = lamb
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lamb * grad_output, None



class ChromTransfer_Base(nn.Module):
    def __init__(self):
        super(ChromTransfer_Base, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=120, kernel_size=20, padding="same")
        self.maxpool1d = nn.MaxPool1d(kernel_size=15, stride=15, padding=5)
        self.lstm = nn.LSTM(input_size=120, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, one_hot_seq):
        # Sequence path
        x = one_hot_seq 
        x = torch.relu(self.conv1d(x))
        x = self.maxpool1d(x)
        
        # Classifier path (bound or unbound)
        classfier = x.permute(0, 2, 1)
        classfier, _ = self.lstm(classfier)
        classfier = classfier[:, -1, :]
        classfier = torch.relu(self.fc1(classfier))
        classfier = self.dropout(classfier)
        classfier = torch.relu(self.fc2(classfier))
        classfier = self.fc3(classfier)
        classifier_output = torch.sigmoid(classfier)
        
        return classifier_output
    
    
class ChromTransfer_Cons(nn.Module):
    def __init__(self):
        super(ChromTransfer_Cons, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=120, kernel_size=20, padding="same")
        self.maxpool1d = nn.MaxPool1d(kernel_size=15, stride=15, padding=5)
        self.lstm = nn.LSTM(input_size=120, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32 + 12, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, one_hot_seq, FUNCODE_data):
        # Sequence path
        x = one_hot_seq 
        x = torch.relu(self.conv1d(x))
        x = self.maxpool1d(x)
        
        # Classifier path (bound or unbound)
        classfier = x.permute(0, 2, 1)
        classfier, _ = self.lstm(classfier)
        classfier = classfier[:, -1, :]
        classfier = torch.relu(self.fc1(torch.cat([classfier, FUNCODE_data], dim=1)))
        classfier = self.dropout(classfier)
        classfier = torch.relu(self.fc2(classfier))
        classfier = self.fc3(classfier)
        classifier_output = torch.sigmoid(classfier)
        
        return classifier_output
    


class ChromTransfer_Reg(nn.Module):
    def __init__(self, cis_feature_num):
        super(ChromTransfer_Reg, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=4, out_channels=120, kernel_size=20, padding="same")
        self.maxpool1d = nn.MaxPool1d(kernel_size=15, stride=15, padding=5)
        self.lstm = nn.LSTM(input_size=120, hidden_size=32, batch_first=True)
        self.fc1 = nn.Linear(32 + 12 + cis_feature_num, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, one_hot_seq, FUNCODE_data, cobindingTFs_data):
        x = one_hot_seq
        x = torch.relu(self.conv1d(x))
        x = self.maxpool1d(x)

        classfier = x.permute(0, 2, 1)
        classfier, _ = self.lstm(classfier)
        classfier = classfier[:, -1, :]
        classfier = torch.relu(self.fc1(torch.cat([classfier, FUNCODE_data, cobindingTFs_data], dim=1)))
        classfier = self.dropout(classfier)
        classfier = torch.relu(self.fc2(classfier))
        classfier = self.fc3(classfier)
        classifier_output = torch.sigmoid(classfier)
        return classifier_output
