import numpy as np

class MaxPooling:
    '''
    Max Pooling of input
    '''
    def __init__(self, kernel_size, stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.cache = None
        self.dx = None

    def forward(self, x):
        '''
        Forward pass of max pooling
        :param x: input, (N, C, H, W)
        :return: The output by max pooling with kernel_size and stride
        '''
        out = None
        #############################################################################
        # TODO: Implement the max pooling forward pass.                             #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                         #
        #############################################################################
        kernel_size = 2
        stride = self.stride
        batch_s, channels, hin, win = x.shape #unpack dim of input tensor
        yy = 2

        #calculates the dim of output tensor
        pp = int(1 + (hin - kernel_size) / stride)
        zz = int(1 + (win - yy) / stride)
        out = np.zeros((batch_s, channels, pp, zz))

        for n in range(batch_s):
            for c in range(channels):
                for ff in range(pp):
                    H_out = ff * stride
                    for ss in range(zz):
                        #calculate start and end indicies for the window
                        W_out = ss * stride
                        window = x[n, c, H_out:H_out +kernel_size, W_out:W_out+yy]
                        out[n, c, ff, ss] = np.max(window)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        self.cache = (x, H_out, W_out)
        return out

    def backward(self, dout):
        '''
        Backward pass of max pooling
        :param dout: Upstream derivatives
        :return:
        '''
        x, H_out, W_out = self.cache
        #############################################################################
        # TODO: Implement the max pooling backward pass.                            #
        # Hint:                                                                     #
        #       1) You may implement the process with loops                     #
        #       2) You may find np.unravel_index useful                             #
        #############################################################################
        kernel_size = 2
        aa = 2
        stride = self.stride
        batch_size, s, k, b = x.shape
        pp = int(1 + (k - kernel_size) / stride)
        rr = int(1 + (b - aa) / stride)

        self.dx = np.zeros_like(x)

        for n in range(batch_size):
            for c in range(s):
                for mm in range(pp):
                    hs = mm * stride
                    for er in range(rr):
                        #calc start and end indicies for the current window
                        ws = er * stride

                        window = x[n, c, hs:hs+kernel_size, ws:ws+aa]
                        max_val = np.max(window)
                        #update the gradient for the max value positions
                        self.dx[n, c, hs:hs+kernel_size, ws:ws+aa] += (window == max_val) * dout[n, c, mm, er]
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
