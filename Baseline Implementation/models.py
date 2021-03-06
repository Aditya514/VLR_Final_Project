import torch
import torch.nn as nn
import torch.nn.functional as F


# Conv NN constructors:
class ConvNN(nn.Module):
    name = ""

    def __init__(self, params):
        """
        The purpose in using this class in the way it built is to make the process of creating CNNs with the ability
        to control its capacity in efficient way (programming efficiency - not time efficiency).
        I found it very useful in constructing experiments. I tried to make this class general as possible.
        :param params: a dictionary with the following attributes:
            capacity influence:
            - channels_lst: lst of the channels sizes. the network complexity is inflected mostly by this parameter
              * for efficiency channels_lst[0] is the number of input channels
            - #FC_Layers: number of fully connected layers
            - extras_blocks_components: in case we want to add layers from the list ["dropout", "max_pool", "batch norm"]
                                        to each block we can do it. Their parameters are attributes of this dict also.
              * notice that if max_pool in extras_blocks_components then we reduce dims using max_pool instead conv
                layer (the conv layer will be with stride 1 and padding)
            - p_dropout: the dropout parameter

            net structure:
            - in_wh: input width and height
        """
        super().__init__()
        self.params = params
        channels_lst = params["channels_lst"]
        extras_block_components = params["extras_blocks_components"]

        assert 2 <= len(channels_lst) <= 5
        conv_layers = []
        for i in range(1, len(channels_lst)):
            """
            Dims calculations: next #channels x (nh-filter_size/2)+1 x (nw-filter_size/2)+1
            """
            filter_size, stride, padding = (4, 2, 1) if "max_pool" not in extras_block_components else (5, 1, 2)
            conv_layers.append(nn.Conv2d(channels_lst[i - 1], channels_lst[i], filter_size, stride, padding, bias=False))
            conv_layers.append(params["activation"]())

            for comp in extras_block_components:
                if comp == "dropout":
                    conv_layers.append(nn.Dropout(params["p_dropout"]))
                if comp == "max_pool":
                    conv_layers.append(nn.MaxPool2d(2, 2))
                if comp == "batch_norm":
                    conv_layers.append(nn.BatchNorm2d(channels_lst[i]))

        out_channels = channels_lst[-1]
        if params["CNN_out_channels"] is not None:
            conv_layers.append(nn.Conv2d(channels_lst[-1], params["CNN_out_channels"], 1))
            conv_layers.append(params["activation"]())
            out_channels = params["CNN_out_channels"]

        self.cnn = nn.Sequential(*conv_layers)

        lin_layers = []
        wh = params["in_wh"] // (2 ** (len(channels_lst) - 1))  # width and height of last layer output
        lin_layer_width = out_channels * (wh ** 2)
        for _ in range(params["#FC_Layers"] - 1):
            lin_layers.append(nn.Linear(lin_layer_width, lin_layer_width))
        lin_layers.append(nn.Linear(lin_layer_width, params["out_size"]))
        self.linear_nn = nn.Sequential(*lin_layers)

        """ we use CE loss so we don't need to apply softmax (for test loss we also use the same CE. for accuracy
            we choose the highest value - this property is saved under softmax)"""

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.view(1, *x.shape)
        assert x.shape[2] == x.shape[3] == self.params["in_wh"]
        assert x.shape[1] == self.params["channels_lst"][0]

        cnn_output = self.cnn(x).view((x.shape[0], -1))
        lin_output = self.linear_nn(cnn_output)
        return lin_output


def create_conv_nn(params):
    return ConvNN(params)


class CNNTrafficSignNet(nn.Module):
    name = "CNN-TrafficSignNet"

    def __init__(self, in_channels=3, out_channels=43):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 50, 3, padding=1, stride=2),  # 32
            nn.ReLU(),
            nn.Dropout(0.05),
            # nn.MaxPool2d(2, stride=2),  # 16
            nn.Conv2d(50, 100, 3, padding=1, stride=2),  # 16
            nn.ReLU(),
            nn.Dropout(0.05),
            # nn.MaxPool2d(2, stride=2),  # 8
            nn.Conv2d(100, 150, 5),  # 4
            nn.ReLU(),
            nn.Conv2d(150, 75, 3, padding=1),  # 4
            nn.ReLU(),
            nn.Conv2d(75, 50, 1),
            nn.ReLU(),
        )

        self.lin = nn.Sequential(
            nn.Linear(50 * 4 * 4, 100),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(100, out_channels)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 50 * 4 * 4)
        x = self.lin(x)
        return x
