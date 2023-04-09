import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self, act_fcn="relu", dropout_rate=0.25):
        super(EEGNet, self).__init__()
        
        act_fcns = {"relu": nn.ReLU(), "elu": nn.ELU(), "leaky_relu": nn.LeakyReLU()}
        self.act_fcn = act_fcns[act_fcn.lower()]

        self.dropout_rate = dropout_rate

        self.first_conv = nn.Sequential(
            nn.Conv2d(
                1,
                16,
                kernel_size=(1, 51),
                stride=(1, 1),
                padding=(0, 25),
                bias=False,
            ),
            nn.BatchNorm2d(16),
        )
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                16,
                32,
                kernel_size=(2, 1),
                stride=(1, 1),
                groups=16,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            self.act_fcn,
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(p=self.dropout_rate),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32),
            self.act_fcn,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p=self.dropout_rate),
        )
        self.classify = nn.Sequential(nn.Linear(736, 2))

    def forward(self, x):
        x = self.first_conv(x)
        x = self.depthwise_conv(x)
        x = self.separable_conv(x)
        x = x.view(x.size(0), -1)
        x = self.classify(x)
        return x
