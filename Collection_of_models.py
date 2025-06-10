import torch
import torch.nn as nn
import torch.nn.functional as F
from Config_PreProc_Train_Param import CONFIG_Model_Training


# ----------------------------
# 3D transformer attention block.
# ----------------------------
class TransformerAttention3D(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dropout=0.1):
        super(TransformerAttention3D, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        # Flatten spatial (and depth) dimensions: token length = D*H*W
        x_flat = x.view(B, C, D * H * W)  # shape [B, C, tokens]
        x_flat = x_flat.permute(2, 0, 1)    # shape [tokens, B, C]
        x_att = self.transformer_encoder(x_flat)  # process through transformer encoder
        # Permute back and reshape to [B, C, D, H, W]
        x_out = x_att.permute(1, 2, 0).view(B, C, D, H, W)
        return x_out


# ----------------------------
# Squeeze-and-Excitation Block
# ----------------------------
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (B, C, D, H, W)
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)  # shape (B, C)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


# ----------------------------
# Residual Upsample Block 3D with SE Attention
# ----------------------------
class Residual_SEAttention_UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, mode='trilinear', use_attention=True, num_groups=8):
        super(Residual_SEAttention_UpsampleBlock3d, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        # Skip connection: if scale factor != 1 or channel sizes differ, we adjust the identity.
        if scale_factor != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        self.use_attention = use_attention
        if self.use_attention:
            self.se = SEBlock(out_channels, reduction=16)

    def forward(self, x):
        identity = self.skip(x)
        out = self.upsample(x)
        out = self.relu(self.gn1(self.conv1(out)))
        out = self.gn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        if self.use_attention:
            out = self.se(out)
        return out


#.<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DL-Models With Normalization-layers >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#............................................................................................................................................#...............................................................................................

#-------------------------------------- Model 4: Light DRR only (Base) -------------------------------------------

class Model_4_DRRSOnly(nn.Module):
    def __init__(self, in_channels_2d, out_slices):
        super(Model_4_DRRSOnly, self).__init__()

        # 2D encoder: process DRRs
        def conv_block_2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                          stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64,  kernel_size=3, stride=2),   # Layer 1
            conv_block_2d(64, 128,   kernel_size=3, stride=2),              # Layer 3
            conv_block_2d(128, 256,  kernel_size=3, stride=2),              # Layer 5
            conv_block_2d(256, 512,  kernel_size=3, stride=2),              # Layer 7
            conv_block_2d(512, 1024, kernel_size=3, stride=2),              # Layer 9
        ])
        # After 5 downsampling layers on a 128x128 input,
        # the final 2D feature map is of size (B,1024,4,4).
        # We split the channel dimension into depth and channels.
        # For example, we can reshape to (B,512,2,4,4) because 512*2=1024.
        # (Here the “2” will be our pseudo-3D depth.)
        
        # Modify the transform layer to handle only the 2D branch output.
        self.transform_layer = nn.Conv3d(512, 512, kernel_size=1)

        # Define a helper upsampling block for the 3D decoder.
        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear',
                              kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                          stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

        # Decoder layers.
        # Note: The original decoder expected an input of (B,1024,2,4,4)
        # coming from the concatenation of the 2D and 3D branches.
        # Now, our input to the decoder is (B,512,2,4,4).
        self.upsample3d_9 = upsample_block_3d(512, 512, scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(512, 512, scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(512, 256, scale_factor=1)    # refinement block
        self.upsample3d_6 = upsample_block_3d(256, 128, scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(128, 64,  scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(64, 32,   scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(32, 1,    scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d,
                                nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d):
        # Process the 2D DRR input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)
        # x_2d shape: (B, 1024, 4, 4)
        # Reshape into a pseudo-3D tensor:
        x_2d_features = x_2d.view(-1, 512, 2, 4, 4)  # splitting 1024 channels into (512, 2)
        # Transform the features using a 3D conv (acts as a bottleneck)
        x_bottleneck = self.transform_layer(x_2d_features)
        # Decoder: progressively upsample the pseudo-3D volume
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck)
        x_upsample3d_8 = self.upsample3d_8(x_upsample3d_9)
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)
        x_upsample3d_6 = self.upsample3d_6(x_upsample3d_7)
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)
        # Remove any extra singleton dimension if needed.
        x_out = x_upsample3d_1.squeeze(0)
        return x_out

#-------------------------------------- Model 4: Light DRR and CT (Base) -------------------------------------------

class Model_4_Base_Simplified_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_4_Base_Simplified_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(64, 128,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(128, 256,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(256, 512, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(512, 1024, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512,    kernel_size=3,  stride=2, padding=1)            # Layer 7 ([1,1024,4,8,8])
        self.conv3d_8 = conv_block_3d(512, 512,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1) 

        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, stride = 1, padding=1):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024, 1024,       scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512, 256,         scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,          scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 512,2,4,4)           
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)    
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)   #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_8 = self.upsample3d_8(x_upsample3d_9)                      #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                      #torch.Size([1, 512, 8, 16, 16])
        x_upsample3d_6 = self.upsample3d_6(x_upsample3d_7)                          
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#-------------------------------------- Model 5: Simplified Base-Skip -----------------------------------

class Model_5_Base_2Skips_Simplified(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_5_Base_2Skips_Simplified, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )


        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 256, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(256, 512,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(512, 1024,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(1024, 2048, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(2048, 4096, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 128, kernel_size=3, stride=2, padding=1)  # Layer 1
        self.conv3d_2 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(256, 512,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(512, 1024,    kernel_size=3,  stride=2, padding=1)            # Layer 7
        self.conv3d_8 = conv_block_3d(1024, 2048,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(4096, 4096, kernel_size=1)  

        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, stride = 1, padding=1):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
            )

        # Decoder layers as individual attributes
        self.upsample3d_9 = upsample_block_3d(4096, 1024,           scale_factor=2)  
        self.upsample3d_8 = upsample_block_3d(1024+1024, 1024,      scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,            scale_factor=1)   
        self.upsample3d_6 = upsample_block_3d(512 + 512, 256,       scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,             scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,              scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,                scale_factor=1)
        #self.final_layer_0 = nn.Conv2d(64, out_slices, kernel_size=3, stride=1, padding=0, bias=False)

        # Call the weight initialization
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 2048, 2, 4, 4)
        x_3d_features = conv3d_8.view(-1, 2048, 2, 4, 4)
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)

        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)  # doubles dims
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)
        x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)
        x_upsample3d_6 = self.upsample3d_6(x_concat_2)
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#------------------------------------- Model 5: Light (Base-Skip) ---------------------------------------

class Model_5_Base_2Skips_Simplified_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_5_Base_2Skips_Simplified_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(64, 128,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(128, 256,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(256, 512, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(512, 1024, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512,    kernel_size=3,  stride=2, padding=1)            # Layer 7 ([1,1024,4,8,8])
        self.conv3d_8 = conv_block_3d(512, 512,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1) 

        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, stride = 1, padding=1):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+512, 1024,  scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512+256, 256,     scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,          scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 512,2,4,4)           
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)    
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)   #torch.Size([1, 1024, 4, 8, 8])
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)               #torch.Size([1, 1536, 4, 8, 8])
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)                          #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                      #torch.Size([1, 512, 8, 16, 16])
        x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)               #torch.Size([1, 768, 8, 16, 16])
        x_upsample3d_6 = self.upsample3d_6(x_concat_2)                          
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#------------------------------------- Model 5: 1Skip-Light (Base-Skip) ---------------------------------

class Model_5_Base_3Skips_Simplified_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_5_Base_3Skips_Simplified_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(64, 128,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(128, 256,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(256, 512, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(512, 1024, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512,    kernel_size=3,  stride=2, padding=1)            # Layer 7 ([1,1024,4,8,8])
        self.conv3d_8 = conv_block_3d(512, 512,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1) 

        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, stride = 1, padding=1):
            return nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride= stride, padding=padding, bias=False),
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+512, 1024,   scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512+256, 256,     scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128 + 64, 64,     scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 512,2,4,4)           
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)    
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)   #torch.Size([1, 1024, 4, 8, 8])
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)               #torch.Size([1, 1536, 4, 8, 8])
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)                          #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                      #torch.Size([1, 512, 8, 16, 16])
        x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)               #torch.Size([1, 768, 8, 16, 16])
        x_upsample3d_6 = self.upsample3d_6(x_concat_2)                          
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_concat_3 = torch.cat((x_upsample3d_4, conv3d_0), dim=1)
        x_upsample3d_2 = self.upsample3d_2(x_concat_3)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#-------------------------------------- Model 6: Simplified architecture (Base-Skip-Res) ----------------

class Model_6_Base_2Skip_Res_Simplified(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_6_Base_2Skip_Res_Simplified, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 256, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(256, 512,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(512, 1024,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(1024, 2048, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(2048, 4096, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 128, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(256, 512,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(512, 1024,    kernel_size=3,  stride=2, padding=1)            # Layer 7
        self.conv3d_8 = conv_block_3d(1024, 2048,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(4096, 4096, kernel_size=1) 

        class ResidualUpsampleBlock3d(nn.Module):
            def __init__(self, in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, padding=1):
                super().__init__()
                self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
                self.conv_block = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),

                )
                
                if scale_factor == 1 and in_channels == out_channels:
                    self.skip = nn.Identity()
                else:
                    self.skip = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                )

            def forward(self, x):
                return self.conv_block(self.upsample(x)) + self.skip(x) 


        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', conv_kernel_size=3, conv_padding=1):
            return nn.Sequential(
            ResidualUpsampleBlock3d(in_channels, out_channels, scale_factor, mode=mode, kernel_size=conv_kernel_size, padding=conv_padding)  
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(4096, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+1024, 1024,  scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512+512, 256,     scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,          scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 2048, 2, 4, 4)
        print(x_2d_features.shape)
        
        x_3d_features = conv3d_8.view(-1, 2048, 2, 4, 4)
        print(x_3d_features.shape)
        
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        print(combined_features.shape)
        
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        print(x_bottleneck_features_transformed.shape)
        

        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)  # doubles dims
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)
        x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)
        x_upsample3d_6 = self.upsample3d_6(x_concat_2)
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#-------------------------------------- Model 6: Light architecture (Base-Skip-Res) ---------------------

class Model_6_Base_2Skip_Res_Simplified_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_6_Base_2Skip_Res_Simplified_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(64, 128,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(128, 256,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(256, 512, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(512, 1024, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512,    kernel_size=3,  stride=2, padding=1)            # Layer 7 ([1,1024,4,8,8])
        self.conv3d_8 = conv_block_3d(512, 512,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1) 

        class ResidualUpsampleBlock3d(nn.Module):
            def __init__(self, in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, padding=1):
                super().__init__()
                self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
                self.conv_block = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),

                )
                
                if scale_factor == 1 and in_channels == out_channels:
                    self.skip = nn.Identity()
                else:
                    self.skip = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                )

            def forward(self, x):
                return self.conv_block(self.upsample(x)) + self.skip(x) 


        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', conv_kernel_size=3, conv_padding=1):
            return nn.Sequential(
            ResidualUpsampleBlock3d(in_channels, out_channels, scale_factor, mode=mode, kernel_size=conv_kernel_size, padding=conv_padding)  
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+512, 1024,  scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512+256, 256,     scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,          scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 512,2,4,4)           
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)    
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)   #torch.Size([1, 1024, 4, 8, 8])
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)               #torch.Size([1, 1536, 4, 8, 8])
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)                          #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                      #torch.Size([1, 512, 8, 16, 16])
        x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)               #torch.Size([1, 768, 8, 16, 16])
        x_upsample3d_6 = self.upsample3d_6(x_concat_2)                          
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#-------------------------------------- Model 6: Light and Simple-ResCon (Base-Skip-Res)-----------------

class Model_6_Base_2Skip_EasyRes_Simplified_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_6_Base_2Skip_EasyRes_Simplified_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(64, 128,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(128, 256,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(256, 512, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(512, 1024, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512,    kernel_size=3,  stride=2, padding=1)            # Layer 7 ([1,1024,4,8,8])
        self.conv3d_8 = conv_block_3d(512, 512,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1) 

        class ResidualUpsampleBlock3d(nn.Module):
            def __init__(self, in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, padding=1):
                super().__init__()
                self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
                self.conv_block = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),

                )
                
                self.skip = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                )

            def forward(self, x):
                return self.conv_block(self.upsample(x)) + self.skip(x) 


        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', conv_kernel_size=3, conv_padding=1):
            return nn.Sequential(
            ResidualUpsampleBlock3d(in_channels, out_channels, scale_factor, mode=mode, kernel_size=conv_kernel_size, padding=conv_padding)  
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+512, 1024,  scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512+256, 256,     scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,          scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 512,2,4,4)           
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)    
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)   #torch.Size([1, 1024, 4, 8, 8])
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)               #torch.Size([1, 1536, 4, 8, 8])
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)                          #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                      #torch.Size([1, 512, 8, 16, 16])
        x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)               #torch.Size([1, 768, 8, 16, 16])
        x_upsample3d_6 = self.upsample3d_6(x_concat_2)                          
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer

#-------------------------------------- Model 6: 1 skip, Light and Simple-ResCon ------------------------


class Model_6_Base_1Skip_EasyRes_Simplified_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_6_Base_1Skip_EasyRes_Simplified_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size = 3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels,affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),  # Layer 1
            conv_block_2d(64, 128,   kernel_size=3,   stride=2),             # Layer 3
            conv_block_2d(128, 256,  kernel_size=3,   stride=2),            # Layer 5
            conv_block_2d(256, 512, kernel_size=3,   stride=2),           # Layer 7
            conv_block_2d(512, 1024, kernel_size=3,   stride=2),           # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128,     kernel_size=3,  stride=2, padding=1)             # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256,     kernel_size=3,  stride=2, padding=1)             # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512,    kernel_size=3,  stride=2, padding=1)            # Layer 7 ([1,1024,4,8,8])
        self.conv3d_8 = conv_block_3d(512, 512,   kernel_size=3,  stride=2, padding=1)           #([1, 2048, 2, 4, 4])

        # Transform layer of concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1) 

        class ResidualUpsampleBlock3d(nn.Module):
            def __init__(self, in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, padding=1):
                super().__init__()
                self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
                self.conv_block = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),

                )
                
                self.skip = nn.Sequential(
                nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                )

            def forward(self, x):
                return self.conv_block(self.upsample(x)) + self.skip(x) 


        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', conv_kernel_size=3, conv_padding=1):
            return nn.Sequential(
            ResidualUpsampleBlock3d(in_channels, out_channels, scale_factor, mode=mode, kernel_size=conv_kernel_size, padding=conv_padding)  
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024,       scale_factor=2)  # doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+512, 1024,  scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512,        scale_factor=1)   # no spatial change, only conv refinement
        self.upsample3d_6 = upsample_block_3d(512, 256,     scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128,         scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64,          scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1,            scale_factor=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:  # Ensure weight exists
                    nn.init.ones_(m.weight)
                if m.bias is not None:  # Ensure bias exists
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):

        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation
        x_2d_features = x_2d.view(-1, 512,2,4,4)           
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)    
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)
        # Decoder using upsampling blocks with skip connections
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)   #torch.Size([1, 1024, 4, 8, 8])
        x_concat_1 = torch.cat((x_upsample3d_9, conv3d_6), dim=1)               #torch.Size([1, 1536, 4, 8, 8])
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)                          #torch.Size([1, 1024, 4, 8, 8])
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                      #torch.Size([1, 512, 8, 16, 16])
        #x_concat_2 = torch.cat((x_upsample3d_7, conv3d_4), dim=1)               #torch.Size([1, 768, 8, 16, 16])
        x_upsample3d_6 = self.upsample3d_6(x_upsample3d_7)                          
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer


#.<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< DL-Models with Attention >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#............................................................................................................................................#

class AttentionBlock3d(nn.Module):
    """
    An attention gate for 3D data that filters encoder features (x)
    based on the gating signal (g) from the decoder.
    """
    def __init__(self, F_g, F_l, F_int):
        """
        Args:
            F_g: Number of channels in the gating signal.
            F_l: Number of channels in the encoder skip connection.
            F_int: Number of intermediate channels.
        """
        super(AttentionBlock3d, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g: gating signal from the decoder
        # x: encoder features (skip connection)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # Multiply the attention coefficients with the encoder features
        return x * psi

    
class Model_6_Base_2Skip_EasyRes_Attention_Light(nn.Module):
    def __init__(self, in_channels_2d, in_channels_3d, out_slices):
        super(Model_6_Base_2Skip_EasyRes_Attention_Light, self).__init__()

        def conv_block_2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm2d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv_layers_2d = nn.ModuleList([
            conv_block_2d(in_channels_2d, 64, kernel_size=3, stride=2),   # Layer 1
            conv_block_2d(64, 128, kernel_size=3, stride=2),               # Layer 3
            conv_block_2d(128, 256, kernel_size=3, stride=2),              # Layer 5
            conv_block_2d(256, 512, kernel_size=3, stride=2),              # Layer 7
            conv_block_2d(512, 1024, kernel_size=3, stride=2),             # Layer 9
        ])

        def conv_block_3d(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            )

        self.conv3d_0 = conv_block_3d(in_channels_3d, 64, kernel_size=3, stride=2, padding=1)      # Layer 1
        self.conv3d_2 = conv_block_3d(64, 128, kernel_size=3, stride=2, padding=1)                 # Layer 3
        self.conv3d_4 = conv_block_3d(128, 256, kernel_size=3, stride=2, padding=1)                # Layer 5
        self.conv3d_6 = conv_block_3d(256, 512, kernel_size=3, stride=2, padding=1)                # Layer 7 ([B,512,?])
        self.conv3d_8 = conv_block_3d(512, 512, kernel_size=3, stride=2, padding=1)                # ([B,512,2,4,4])

        # Transform layer for concatenated features
        self.transform_layer = nn.Conv3d(1024, 1024, kernel_size=1)

        # Define a residual upsample block for 3D data
        class ResidualUpsampleBlock3d(nn.Module):
            def __init__(self, in_channels, out_channels, scale_factor, mode='trilinear', kernel_size=3, padding=1):
                super().__init__()
                self.upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True)
                self.conv_block = nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
                )
                self.skip = nn.Sequential(
                    nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=True),
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
                )

            def forward(self, x):
                return self.conv_block(self.upsample(x)) + self.skip(x)

        def upsample_block_3d(in_channels, out_channels, scale_factor, mode='trilinear', conv_kernel_size=3, conv_padding=1):
            return nn.Sequential(
                ResidualUpsampleBlock3d(in_channels, out_channels, scale_factor, mode=mode,
                                        kernel_size=conv_kernel_size, padding=conv_padding)
            )

        # Decoder layers using upsampling blocks
        self.upsample3d_9 = upsample_block_3d(1024, 1024, scale_factor=2)   # Doubles spatial dims
        self.upsample3d_8 = upsample_block_3d(1024+512, 1024, scale_factor=2)
        self.upsample3d_7 = upsample_block_3d(1024, 512, scale_factor=1)     # Refinement (no spatial change)
        self.upsample3d_6 = upsample_block_3d(512, 256, scale_factor=2)
        self.upsample3d_4 = upsample_block_3d(256, 128, scale_factor=2)
        self.upsample3d_2 = upsample_block_3d(128, 64, scale_factor=2)
        self.upsample3d_1 = upsample_block_3d(64, 1, scale_factor=1)

        # Attention blocks for the two skip connections:
        # First skip: gating signal from upsample3d_9 (1024 channels) and skip from conv3d_6 (512 channels)
        self.attention1 = AttentionBlock3d(F_g=1024, F_l=512, F_int=256)
        # Second skip: gating signal from upsample3d_7 (512 channels) and skip from conv3d_4 (256 channels)
        self.attention2 = AttentionBlock3d(F_g=512, F_l=256, F_int=128)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_2d, x_3d):
        # Process 2D input
        for conv in self.conv_layers_2d:
            x_2d = conv(x_2d)

        # Process 3D input (encoder branch)
        conv3d_0 = self.conv3d_0(x_3d)
        conv3d_2 = self.conv3d_2(conv3d_0)
        conv3d_4 = self.conv3d_4(conv3d_2)
        conv3d_6 = self.conv3d_6(conv3d_4)
        conv3d_8 = self.conv3d_8(conv3d_6)

        # Adjust dimensions for concatenation with 2D branch
        x_2d_features = x_2d.view(-1, 512, 2, 4, 4)
        x_3d_features = conv3d_8.view(-1, 512, 2, 4, 4)
        combined_features = torch.cat((x_2d_features, x_3d_features), dim=1)
        x_bottleneck_features_transformed = self.transform_layer(combined_features)

        # Decoder with attention-gated skip connections
        # First skip connection:
        x_upsample3d_9 = self.upsample3d_9(x_bottleneck_features_transformed)  # [B,1024,4,8,8]
        att_conv3d_6 = self.attention1(g=x_upsample3d_9, x=conv3d_6)           # Apply attention on conv3d_6
        x_concat_1 = torch.cat((x_upsample3d_9, att_conv3d_6), dim=1)            # [B,1024+512,4,8,8]
        x_upsample3d_8 = self.upsample3d_8(x_concat_1)                           # [B,1024,4,8,8]

        # Second skip connection:
        x_upsample3d_7 = self.upsample3d_7(x_upsample3d_8)                       # [B,512,8,16,16]
        att_conv3d_4 = self.attention2(g=x_upsample3d_7, x=conv3d_4)             # Apply attention on conv3d_4
        x_concat_2 = torch.cat((x_upsample3d_7, att_conv3d_4), dim=1)            # [B,512+256,8,16,16]
        x_upsample3d_6 = self.upsample3d_6(x_upsample3d_7)                           # [B,256,16,32,32]
        x_upsample3d_4 = self.upsample3d_4(x_upsample3d_6)
        x_upsample3d_2 = self.upsample3d_2(x_upsample3d_4)
        x_upsample3d_1 = self.upsample3d_1(x_upsample3d_2)

        x_to_final_layer = x_upsample3d_1.squeeze(0)
        return x_to_final_layer
