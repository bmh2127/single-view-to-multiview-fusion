import torch
import torch.nn as nn

import logging


class DoubleConv3D(nn.Module):
    """
    Double 3D convolution block with batch normalization and ReLU activations.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class Down3D(nn.Module):
    """
    Downsampling block with maxpooling followed by double convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Down3D, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.mpconv(x)


class Up3D(nn.Module):
    """
    Upsampling block with either transposed convolution or bilinear upsampling,
    followed by concatenation with skip connection and double convolution.
    """
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(Up3D, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels)
        else:
            # Use transposed convolution
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Ensure x1 has the same size as x2
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        # Pad x1 if needed
        x1 = nn.functional.pad(
            x1, 
            [diff_x // 2, diff_x - diff_x // 2,
             diff_y // 2, diff_y - diff_y // 2,
             diff_z // 2, diff_z - diff_z // 2]
        )
        
        # Concatenate along the channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net model for volumetric image processing.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()
        
        features = init_features
        
        self.encoder1 = DoubleConv3D(in_channels, features)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.encoder4 = DoubleConv3D(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv3D(features * 8, features * 16)
        
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv3D(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv3D(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv3D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv3D(features * 2, features)
        
        self.final_conv = nn.Conv3d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        # Handle any size mismatch
        if dec4.size()[2:] != enc4.size()[2:]:
            dec4 = nn.functional.interpolate(dec4, size=enc4.size()[2:], mode='trilinear', align_corners=True)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        
        dec3 = self.upconv3(dec4)
        if dec3.size()[2:] != enc3.size()[2:]:
            dec3 = nn.functional.interpolate(dec3, size=enc3.size()[2:], mode='trilinear', align_corners=True)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        
        dec2 = self.upconv2(dec3)
        if dec2.size()[2:] != enc2.size()[2:]:
            dec2 = nn.functional.interpolate(dec2, size=enc2.size()[2:], mode='trilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        
        dec1 = self.upconv1(dec2)
        if dec1.size()[2:] != enc1.size()[2:]:
            dec1 = nn.functional.interpolate(dec1, size=enc1.size()[2:], mode='trilinear', align_corners=True)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        
        # Final output
        return self.final_conv(dec1)


class PhysicsInformedUNet3D(UNet3D):
    """
    Physics-informed variant of the 3D U-Net model.
    This extends the basic UNet3D with additional layers for physics constraints.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(PhysicsInformedUNet3D, self).__init__(in_channels, out_channels, init_features)
        
        # Additional physics-related layers
        self.physics_conv1 = nn.Conv3d(init_features, init_features, kernel_size=3, padding=1)
        self.physics_relu = nn.ReLU(inplace=True)
        self.physics_conv2 = nn.Conv3d(init_features, init_features, kernel_size=3, padding=1)
        self.physics_combine = nn.Conv3d(init_features*2, init_features, kernel_size=1)
    
    def forward(self, x):
        # Base UNet forward
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        if dec4.size()[2:] != enc4.size()[2:]:
            dec4 = nn.functional.interpolate(dec4, size=enc4.size()[2:], mode='trilinear', align_corners=True)
        dec4 = self.decoder4(torch.cat((dec4, enc4), dim=1))
        
        dec3 = self.upconv3(dec4)
        if dec3.size()[2:] != enc3.size()[2:]:
            dec3 = nn.functional.interpolate(dec3, size=enc3.size()[2:], mode='trilinear', align_corners=True)
        dec3 = self.decoder3(torch.cat((dec3, enc3), dim=1))
        
        dec2 = self.upconv2(dec3)
        if dec2.size()[2:] != enc2.size()[2:]:
            dec2 = nn.functional.interpolate(dec2, size=enc2.size()[2:], mode='trilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat((dec2, enc2), dim=1))
        
        dec1 = self.upconv1(dec2)
        if dec1.size()[2:] != enc1.size()[2:]:
            dec1 = nn.functional.interpolate(dec1, size=enc1.size()[2:], mode='trilinear', align_corners=True)
        dec1 = self.decoder1(torch.cat((dec1, enc1), dim=1))
        
        # Physics-informed processing
        physics_features = self.physics_conv1(dec1)
        physics_features = self.physics_relu(physics_features)
        physics_features = self.physics_conv2(physics_features)
        
        # Combine with standard features 
        combined = torch.cat([dec1, physics_features], dim=1)
        enhanced = self.physics_combine(combined)
        
        # Final output
        return self.final_conv(enhanced)
        