import torch
import torch.nn as nn


class MyModel(nn.Module):
    # You can use pre-existing models but change layers to recieve full credit.
    def __init__(self):
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize the network weights                                      #
        #############################################################################
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  #3 input channels, 32 output, kernel size of 3
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(64), 

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128), #further increases depth to 128

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),

            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024), 
            nn.ReLU(),
            nn.Dropout(0.5),  #dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  #output size is 10 because 10-classification problem
        )
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        outs = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        outs = self.model(x)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return outs