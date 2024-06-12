import torch 
import math
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, is_res=False):
        super(ResidualBlock, self).__init__()
        self.is_res = is_res
        self.same_channels = in_channels == out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        self.handle_channel = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x_out = self.conv1(x)
        x_out = self.conv2(x_out)

        if self.is_res:
            if self.same_channels:  
                x_out += x
            else:   
                x_out = x_out + self.handle_channel(x)
            return x_out
        else:
            return x_out
            
class DownwardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownwardBlock, self).__init__()
        layers = [
            ResidualBlock(in_channels, out_channels, is_res=True),
            ResidualBlock(out_channels, out_channels),
            nn.MaxPool2d(kernel_size=2)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UpwardBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpwardBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.res_block = ResidualBlock(in_channels, out_channels)

        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 2, 2),
            ResidualBlock(out_channels, out_channels, is_res=True),
            ResidualBlock(out_channels, out_channels)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_connection):
        return self.model(torch.cat([x, skip_connection], dim=1))

class convFCEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(convFCEmbedding, self).__init__()
        self.in_dim = in_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_dim, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.GELU(),
            # nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            # nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            # nn.GELU(),
            # nn.MaxPool2d(2),
            # nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(128),
            # nn.GELU(),
            # nn.MaxPool2d(2),

        )

        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv(x) 
        # print(x.shape)
        # x = self.fc(x)
        # print('--------')
        # x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return x

class FCEmbedding(nn.Module):
    """Fully Connected Embedding Layer"""
    def __init__(self, input_dim, hidden_dim):
        super(FCEmbedding, self).__init__()
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(-1, self.input_dim)
        return self.model(x)[:, :, None, None]
    
class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x, context):
        N, C, H, W = x.size()
        x = x.flatten(2).permute(2, 0, 1)  # (N, C, H, W) -> (H*W, N, C)
        context = context.flatten(2).permute(2, 0, 1)  # (N, C, H, W) -> (H*W, N, C)
        attn_output, _ = self.multihead_attn(x, context, context)
        attn_output = attn_output.permute(1, 2, 0).view(N, C, H, W)  # (H*W, N, C) -> (N, C, H, W)
        return attn_output
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000) / half_dim))
        emb = t.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class CrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x, context):
        N, C, H, W = x.size()
        x = x.flatten(2).permute(2, 0, 1)  # (N, C, H, W) -> (H*W, N, C)
        context = context.flatten(2).permute(2, 0, 1)  # (N, C, H, W) -> (H*W, N, C)
        attn_output, _ = self.multihead_attn(x, context, context)
        attn_output = attn_output.permute(1, 2, 0).view(N, C, H, W)  # (H*W, N, C) -> (N, C, H, W)
        return attn_output

class ContextUNet(nn.Module):
    def __init__(self, in_channels=1, height=128, width=128, n_feat=16, n_cfeat=32, n_downs=4):
        super(ContextUNet, self).__init__()
        self.in_channels = in_channels
        self.height = height
        self.width = width
        self.n_feat = n_feat
        self.n_cfeat = n_cfeat
        self.n_downs = n_downs

        # Define initial convolution
        self.init_conv = ResidualBlock(in_channels, n_feat, True)
        
        # Define downward unet blocks
        self.down_blocks = nn.ModuleList()
        for i in range(n_downs):
            self.down_blocks.append(DownwardBlock(2**i * n_feat, 2**(i+1) * n_feat))
        
        # Define at the center layers
        self.to_vec = nn.Sequential(
            nn.AvgPool2d((height//2**len(self.down_blocks), width//2**len(self.down_blocks))), 
            nn.GELU())
        
        self.up0 = nn.Sequential(
            nn.ConvTranspose2d(
                2**n_downs * n_feat, 
                2**n_downs * n_feat, 
                (height // 2**len(self.down_blocks), width // 2**len(self.down_blocks))),
            nn.GroupNorm(8, 2**n_downs * n_feat),
            nn.GELU()
        )
        
        # Define upward unet blocks
        self.up_blocks = nn.ModuleList()
        for i in range(n_downs, 0, -1):
            self.up_blocks.append(UpwardBlock(2**(i+1) * n_feat, 2**(i-1) * n_feat))

        # Define final convolutional layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(2 * n_feat, n_feat, 3, 1, 1),
            nn.GroupNorm(8, n_feat),
            nn.GELU(),
            nn.Conv2d(n_feat, in_channels, 1, 1)
        )

        ### original version
        self.timeembs = nn.ModuleList([FCEmbedding(1, 2**i*n_feat) for i in range(n_downs, 0, -1)])
        self.contextembs = nn.ModuleList([convFCEmbedding(2**(i-1)*n_feat, 2**i*n_feat) for i in range(n_downs, 0, -1)])
        # self.contextembs = nn.ModuleList([convFCEmbedding(i, 2**i*n_feat) for i in range(n_downs, 0, -1)])
        # 32, 4*32=128
        # 32, 2*32=64

        # self.contextembs = nn.ModuleList([FCEmbedding(2**(i+1) * n_feat, 2**(i-1) * n_feat) for i in range(n_downs, 0, -1)])

        ### add cross attention
        # Define time & context embedding blocks 
        # self.timeembs = TimeEmbedding(n_feat) # nn.ModuleList([FCEmbedding(1, 2**i * n_feat) for i in range(n_downs, 0, -1)])
        # self.contextembs = nn.ModuleList([FCEmbedding(n_cfeat, 2**i * n_feat) for i in range(n_downs, 0, -1)])

        # # Define cross attention layer
        # self.cross_attention = CrossAttention(d_model=2**n_downs * n_feat, nhead=8)

    def forward(self, x, t, condition_image):
        x = self.init_conv(x)
        downs = []
        for i, down_block in enumerate(self.down_blocks):
            if i == 0: 
                downs.append(down_block(x)) # this line
            else: 
                downs.append(down_block(downs[-1])) 
        up = self.up0(self.to_vec(downs[-1]))

        # Add cross attention 
        # this line
        # target_emb = self.contextembs[0](condition_image).view(condition_image.size(0), -1, 1, 1)  # Reshape context embedding to match feature map size
        # up = self.cross_attention(up, target_emb)

        condition_feature = []
        for i, layer in enumerate(self.contextembs):
            condition_image = layer(condition_image)
            condition_feature.append(condition_image)
            print(condition_image.shape)
        return 
        
        for i, (up_block, down, contextemb, timeemb) in enumerate(zip(self.up_blocks, downs[::-1], self.contextembs, self.timeembs)):
            # print("\nup:", up.shape) # 32, 128, 16, 16
            # print("time:", timeemb(t).shape) # 32, 128, 1, 1
            # print("emb:", contextemb(condition_image).shape)
            # print('---------------')
            up = up_block(up * contextemb(condition_image) + timeemb(t), down) # fix to this line
            up = up_block(up + timeemb(t), down)
        # print("final, up:", up.shape)
        # print("final, down:", downs[0].shape)
        return self.final_conv(torch.cat([up, x], axis=1))

