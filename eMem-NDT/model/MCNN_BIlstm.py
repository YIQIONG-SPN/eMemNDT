import torch
import torch.nn as nn



class MyModel(nn.Module):
    def __init__(self, num_classes):
        super(MyModel, self).__init__()


        self.lstm = nn.LSTM(input_size=4, hidden_size=64, num_layers=2, batch_first=True, bidirectional=True)


        self.conv1 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=5, out_channels=32, kernel_size=4)
        self.fc1 = nn.Linear(in_features=96, out_features=32)


        self.out_layer = nn.Linear(in_features=32 + 64 * 2, out_features=num_classes)

    def forward(self, x):

        out_lstm, _ = self.lstm(x)


        x_conv1 = torch.relu(self.conv1(x))
        x_conv2 = torch.relu(self.conv2(x))
        x_conv3 = torch.relu(self.conv3(x))

        x_pool1, _ = torch.max(x_conv1, dim=-1)
        x_pool2, _ = torch.max(x_conv2, dim=-1)
        x_pool3, _ = torch.max(x_conv3, dim=-1)

        out_mscnn = torch.cat((x_pool1, x_pool2, x_pool3), dim=1)
        out_mscnn = self.fc1(out_mscnn)


        out = torch.cat((out_lstm[:, -1, :], out_mscnn), dim=1)


        out = self.out_layer(out)

        return out