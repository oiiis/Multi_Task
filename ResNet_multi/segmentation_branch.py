import torch
import torch.nn as nn

class DepthwiseSeparableConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConvBlock3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class SegmentationBranch3D(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(SegmentationBranch3D, self).__init__()
        self.upconv1 = nn.ConvTranspose3d(input_channels, 256, kernel_size=2, stride=2)
        self.ds_block1 = DepthwiseSeparableConvBlock3D(256 + 512, 256)

        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.ds_block2 = DepthwiseSeparableConvBlock3D(128 + 256, 128)

        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.ds_block3 = DepthwiseSeparableConvBlock3D(64 + 128, 64)

        self.upconv4 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.ds_block4 = DepthwiseSeparableConvBlock3D(32 + 64, 32)

        self.final_conv = nn.Conv3d(32, num_classes, kernel_size=1)

    def forward(self, x, encoder_outputs):
        # Ensure there are enough encoder outputs
        if len(encoder_outputs) < 4:
            raise ValueError(f"Expected at least 4 encoder outputs, but got {len(encoder_outputs)}")

        for idx, output in enumerate(encoder_outputs):
            if not torch.isfinite(output).all():
                raise ValueError(f"Invalid values found in encoder output at index {idx}")

        x = self.upconv1(x)
        print(f"Shape after upconv1: {x.shape}")
        print(f"Encoder output shape at index 3: {encoder_outputs[3].shape}")
        x = self._match_size(x, encoder_outputs[3])
        x = torch.cat((x, encoder_outputs[3]), dim=1)  # Skip connection from encoder
        x = self.ds_block1(x)
        print(f"Shape after ds_block1: {x.shape}")

        x = self.upconv2(x)
        print(f"Shape after upconv2: {x.shape}")
        print(f"Encoder output shape at index 2: {encoder_outputs[2].shape}")
        x = self._match_size(x, encoder_outputs[2])
        x = torch.cat((x, encoder_outputs[2]), dim=1)  # Skip connection from encoder
        x = self.ds_block2(x)
        print(f"Shape after ds_block2: {x.shape}")

        x = self.upconv3(x)
        print(f"Shape after upconv3: {x.shape}")
        print(f"Encoder output shape at index 1: {encoder_outputs[1].shape}")
        x = self._match_size(x, encoder_outputs[1])
        x = torch.cat((x, encoder_outputs[1]), dim=1)  # Skip connection from encoder
        x = self.ds_block3(x)
        print(f"Shape after ds_block3: {x.shape}")

        x = self.upconv4(x)
        print(f"Shape after upconv4: {x.shape}")
        print(f"Encoder output shape at index 0: {encoder_outputs[0].shape}")
        x = self._match_size(x, encoder_outputs[0])
        x = torch.cat((x, encoder_outputs[0]), dim=1)  # Skip connection from encoder
        x = self.ds_block4(x)
        print(f"Shape after ds_block4: {x.shape}")

        x = self.final_conv(x)
        print(f"Shape after final_conv: {x.shape}")
        return x

    def _match_size(self, x, target):
        # Ensure the input and target have the same spatial dimensions
        target_size = target.shape[2:]
        input_size = x.shape[2:]
        if input_size != target_size:
            diff_depth = target_size[0] - input_size[0]
            diff_height = target_size[1] - input_size[1]
            diff_width = target_size[2] - input_size[2]

            pad_depth = (diff_depth // 2, diff_depth - diff_depth // 2)
            pad_height = (diff_height // 2, diff_height - diff_height // 2)
            pad_width = (diff_width // 2, diff_width - diff_width // 2)

            x = nn.functional.pad(x, pad_width + pad_height + pad_depth)
        return x
