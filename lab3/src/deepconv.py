import torch.nn as nn


class DeepConvNet(nn.Module):
    def __init__(self, act_fcn="relu", dropout_rate=0.5):
        super(DeepConvNet, self).__init__()
        act_fcns = {"relu": nn.ReLU(), "elu": nn.ELU(), "leaky_relu": nn.LeakyReLU()}
        self.act_fcn = act_fcns[act_fcn.lower()]

        self.dropout_rate = dropout_rate

        channels = (1, 25, 25, 50, 100, 200)
        kernel_sizes = ((1, 5), (2, 1), (1, 5), (1, 5), (1, 5))
        self.conv_layers = nn.ModuleList()
        conv_layer = nn.Conv2d(channels[0], channels[1], kernel_sizes[0])
        self.conv_layers.append(conv_layer)
        for i in range(1, len(channels) - 1):
            conv_layer = nn.Sequential(
                nn.Conv2d(channels[i], channels[i + 1], kernel_sizes[i]),
                nn.BatchNorm2d(channels[i + 1]),
                self.act_fcn,
                nn.MaxPool2d(kernel_size=(1, 2)),
                nn.Dropout(p=self.dropout_rate),
            )
            self.conv_layers.append(conv_layer)
        self.classify = nn.Sequential(nn.Linear(8600, 2))

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x
