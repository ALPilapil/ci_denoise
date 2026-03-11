import torch
import torch.nn as nn

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
        nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
        nn.ELU(inplace=True),
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(19, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)

        self.up_trans_1 = nn.ConvTranspose1d(
            in_channels=1024, 
            out_channels=512,
            kernel_size=2,
            stride=2)
        
        self.up_conv_1 = double_conv(1024, 512)

        self.up_trans_2 = nn.ConvTranspose1d(
            in_channels=512, 
            out_channels=256,
            kernel_size=2,
            stride=2)
        
        self.up_conv_2 = double_conv(512, 256)

        self.up_trans_3 = nn.ConvTranspose1d(
            in_channels=256, 
            out_channels=128,
            kernel_size=2,
            stride=2)
        
        self.up_conv_3 = double_conv(256, 128)

        self.up_trans_4 = nn.ConvTranspose1d(
            in_channels=128, 
            out_channels=64,
            kernel_size=2,
            stride=2)
        
        self.up_conv_4 = double_conv(128, 64)

        self.out = nn.Conv1d(
            in_channels=64,
            out_channels=19, # can adjust this based on num objects to segment 
            kernel_size=1,
        )
        

    def forward(self, image):
        # expected dim: bs, c, timepoints
        # encoder
        x1 = self.down_conv_1(image) #
        x2 = self.max_pool(x1)

        x3 = self.down_conv_2(x2) #
        x4 = self.max_pool(x3)

        x5 = self.down_conv_3(x4) #
        x6 = self.max_pool(x5)

        x7 = self.down_conv_4(x6) #
        x8 = self.max_pool(x7)

        x9 = self.down_conv_5(x8)
        # notice no max pooling here

        # decoder
        x = self.up_trans_1(x9)
        x = self.up_conv_1(torch.cat([x7, x], 1))
        
        x = self.up_trans_2(x)
        x = self.up_conv_2(torch.cat([x5, x], 1))

        x = self.up_trans_3(x)
        x = self.up_conv_3(torch.cat([x3, x], 1))

        x = self.up_trans_4(x)
        x = self.up_conv_4(torch.cat([x1, x], 1))

        x = self.out(x)
        print(x.size())
        return x
        

if __name__ == "__main__":
    # fake image
    # this will still work with any image dim 
    epoch = torch.rand((1,19,6000))
    model = UNet()
    print(model(image))


    
