import torch
import torch.nn as nn

class TwoLayerNet(nn.Module):
    def __init__(self, input_dim, hidden_size, num_classes):
        '''
        :param input_dim: input feature dimension
        :param hidden_size: hidden dimension
        :param num_classes: total number of classes
        '''
        super(TwoLayerNet, self).__init__()
        #############################################################################
        # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
        #############################################################################
        self.fcc1=nn.Linear(input_dim,hidden_size)
        self.sigmoid=nn.Sigmoid()
        self.fcc2=nn.Linear(hidden_size, num_classes) #no softmax bc already used in cross entropy
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, x):
        out = None
        #############################################################################
        # TODO: Implement forward pass of the network                               #
        #############################################################################
        one=self.fcc1(x.view(x.shape[0],-1)) #double check
        two=self.sigmoid(one)
        out=self.fcc2(two) #same reason as above. no softmax bc cross-entropy already has softmax
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out