import torch
import torch.nn as nn

class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        #############################################################################
        # TODO: Initialize the Vanilla CNN                                          #
        #       Conv: 7x7 kernel, stride 1 and padding 0                            #
        #       Max Pooling: 2x2 kernel, stride 2                                   #
        #############################################################################
        self.fc1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7, stride = 1, padding = 0)
        self.rl=nn.ReLU(inplace=True)
        self.pl=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc2=nn.Linear(in_features=5408,out_features=10)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################


    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        one =self.fc1(x) #first conv layer
        two=self.rl(one)
        three = self.pl(two)
        four = three.view(three.size(0), -1)
        outs=self.fc2(four)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return outs