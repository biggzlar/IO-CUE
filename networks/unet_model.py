import torch
import torch.nn as nn
from networks.unet_components import DepthwiseSeparableConv2d, StridedConvBlock, EfficientUpsample, OptimizedHead


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, activation=torch.nn.ReLU):
        super(UNet, self).__init__()
        self.opts = locals().copy()
        del self.opts['self']
        
        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(drop_prob)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(drop_prob)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(drop_prob)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(drop_prob)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upconv6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upconv7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.heads = nn.ModuleList()
        for c in out_channels:
            head = torch.nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(32, c, kernel_size=1)
            )
            self.heads.append(head)        
        
        self.activation = activation()
        
    def forward(self, x):
        # Encoder
        conv1 = self.activation(self.conv1_1(x))
        conv1 = self.activation(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        pool1 = self.drop1(pool1)
        
        conv2 = self.activation(self.conv2_1(pool1))
        conv2 = self.activation(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        pool2 = self.drop2(pool2)
        
        conv3 = self.activation(self.conv3_1(pool2))
        conv3 = self.activation(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)
        pool3 = self.drop3(pool3)
        
        conv4 = self.activation(self.conv4_1(pool3))
        conv4 = self.activation(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)
        pool4 = self.drop4(pool4)
        
        conv5 = self.activation(self.conv5_1(pool4))
        conv5 = self.activation(self.conv5_2(conv5))
        
        # Decoder with skip connections
        up_conv5 = self.upconv5(conv5)
        # up_conv5 = F.interpolate(up_conv5, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        concat6 = torch.cat([up_conv5, conv4], dim=1)
        conv6 = self.activation(self.conv6_1(concat6))
        conv6 = self.activation(self.conv6_2(conv6))
        
        up_conv6 = self.upconv6(conv6)
        # up_conv6 = F.interpolate(up_conv6, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        concat7 = torch.cat([up_conv6, conv3], dim=1)
        conv7 = self.activation(self.conv7_1(concat7))
        conv7 = self.activation(self.conv7_2(conv7))
        
        up_conv7 = self.upconv7(conv7)
        # up_conv7 = F.interpolate(up_conv7, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        concat8 = torch.cat([up_conv7, conv2], dim=1)
        conv8 = self.activation(self.conv8_1(concat8))
        conv8 = self.activation(self.conv8_2(conv8))
        
        up_conv8 = self.upconv8(conv8)
        # up_conv8 = F.interpolate(up_conv8, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        concat9 = torch.cat([up_conv8, conv1], dim=1)
        
        outputs = torch.concat([head(concat9) for head in self.heads], dim=1)        
        
        return outputs


class MergerUNet(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, activation=torch.nn.ReLU):
        super(MergerUNet, self).__init__()
        self.opts = locals().copy()
        del self.opts['self']
        
        # Encoder
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(drop_prob)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(drop_prob)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(drop_prob)
        
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop4 = nn.Dropout2d(drop_prob)
        
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        
        # Decoder
        self.upconv5 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        self.upconv6 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv7_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.upconv7 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv8_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv8_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.upconv8 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.heads = nn.ModuleList()
        for c in out_channels:
            head = torch.nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(32, c, kernel_size=1)
            )
            self.heads.append(head)        
        
        self.activation = activation()
        
    def forward(self, x):
        if x.shape[1] > 3:
            x_, y_ = torch.split(x, [3, 1], dim=1)
            x_ = torch.concat([x_, torch.zeros_like(y_)], dim=1)
            conv1, conv2, conv3, conv4, conv5 = self.forward_encoder(x_)
            x_ = self.forward_decoder(conv1, conv2, conv3, conv4, conv5)

            conv1, conv2, conv3, conv4, conv5 = self.forward_encoder(x)
            x = self.forward_decoder(conv1, conv2, conv3, conv4, conv5)
            embeddings = [x_, x]
            
        else:
            conv1, conv2, conv3, conv4, conv5 = self.forward_encoder(x)
            x = self.forward_decoder(conv1, conv2, conv3, conv4, conv5)
            embeddings = [x]
        
        outputs = []
        for h in embeddings:
            outs = torch.concat([head(h) for head in self.heads], dim=1) 
            outputs.append(outs)
        
        output = torch.concat(outputs, dim=1)
        # outputs = torch.concat([head(h[i]) for i, head in enumerate(self.heads)], dim=1)        
        
        return output

    def forward_encoder(self, x):
        # Encoder
        conv1 = self.activation(self.conv1_1(x))
        conv1 = self.activation(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        pool1 = self.drop1(pool1)
        
        conv2 = self.activation(self.conv2_1(pool1))
        conv2 = self.activation(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        pool2 = self.drop2(pool2)
        
        conv3 = self.activation(self.conv3_1(pool2))
        conv3 = self.activation(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)
        pool3 = self.drop3(pool3)
        
        conv4 = self.activation(self.conv4_1(pool3))
        conv4 = self.activation(self.conv4_2(conv4))
        pool4 = self.pool4(conv4)
        pool4 = self.drop4(pool4)
        
        conv5 = self.activation(self.conv5_1(pool4))
        conv5 = self.activation(self.conv5_2(conv5))

        return conv1, conv2, conv3, conv4, conv5
    

    def forward_decoder(self, conv1, conv2, conv3, conv4, conv5):
        # Decoder with skip connections
        up_conv5 = self.upconv5(conv5)
        # up_conv5 = F.interpolate(up_conv5, size=conv4.shape[2:], mode='bilinear', align_corners=False)
        concat6 = torch.cat([up_conv5, conv4], dim=1)
        conv6 = self.activation(self.conv6_1(concat6))
        conv6 = self.activation(self.conv6_2(conv6))
        
        up_conv6 = self.upconv6(conv6)
        # up_conv6 = F.interpolate(up_conv6, size=conv3.shape[2:], mode='bilinear', align_corners=False)
        concat7 = torch.cat([up_conv6, conv3], dim=1)
        conv7 = self.activation(self.conv7_1(concat7))
        conv7 = self.activation(self.conv7_2(conv7))
        
        up_conv7 = self.upconv7(conv7)
        # up_conv7 = F.interpolate(up_conv7, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        concat8 = torch.cat([up_conv7, conv2], dim=1)
        conv8 = self.activation(self.conv8_1(concat8))
        conv8 = self.activation(self.conv8_2(conv8))
        
        up_conv8 = self.upconv8(conv8)
        # up_conv8 = F.interpolate(up_conv8, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        concat9 = torch.cat([up_conv8, conv1], dim=1)
        
        return concat9

class BabyUNet(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, activation=torch.nn.ReLU):
        super(BabyUNet, self).__init__()
        self.opts = locals().copy()
        del self.opts['self']
        
        # Encoder - reduced to 3 blocks
        self.conv1_1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(drop_prob)
        
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(drop_prob)
        
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Decoder - reduced to 2 blocks
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.heads = nn.ModuleList()
        for c in out_channels:
            head = torch.nn.Sequential(
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(32, c, kernel_size=1)
            )
            self.heads.append(head)        
        
        self.activation = activation()
        
    def forward(self, x):
        # Encoder
        conv1 = self.activation(self.conv1_1(x))
        conv1 = self.activation(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        pool1 = self.drop1(pool1)
        
        conv2 = self.activation(self.conv2_1(pool1))
        conv2 = self.activation(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        pool2 = self.drop2(pool2)
        
        conv3 = self.activation(self.conv3_1(pool2))
        conv3 = self.activation(self.conv3_2(conv3))
        
        # Decoder with skip connections
        up_conv3 = self.upconv3(conv3)
        concat4 = torch.cat([up_conv3, conv2], dim=1)
        conv4 = self.activation(self.conv4_1(concat4))
        conv4 = self.activation(self.conv4_2(conv4))
        
        up_conv4 = self.upconv4(conv4)
        concat5 = torch.cat([up_conv4, conv1], dim=1)
        
        outputs = torch.concat([head(concat5) for head in self.heads], dim=1)        
        
        return outputs


class MediumUNet(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, activation=torch.nn.ReLU):
        super(MediumUNet, self).__init__()
        self.opts = locals().copy()
        del self.opts['self']
        
        # Encoder - 4 blocks with reduced channels
        self.conv1_1 = nn.Conv2d(in_channels, 24, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(drop_prob)
        
        self.conv2_1 = nn.Conv2d(24, 48, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop2 = nn.Dropout2d(drop_prob)
        
        self.conv3_1 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop3 = nn.Dropout2d(drop_prob)
        
        self.conv4_1 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        
        # Decoder - 3 blocks
        self.upconv4 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.conv5_1 = nn.Conv2d(192, 96, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(96, 96, kernel_size=3, padding=1)
        
        self.upconv5 = nn.ConvTranspose2d(96, 48, kernel_size=2, stride=2)
        self.conv6_1 = nn.Conv2d(96, 48, kernel_size=3, padding=1)
        self.conv6_2 = nn.Conv2d(48, 48, kernel_size=3, padding=1)
        
        self.upconv6 = nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2)
        
        self.heads = nn.ModuleList()
        for c in out_channels:
            head = torch.nn.Sequential(
                nn.Conv2d(48, 24, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(24, 24, kernel_size=3, padding=1),
                activation(),
                nn.Conv2d(24, c, kernel_size=1)
            )
            self.heads.append(head)        
        
        self.activation = activation()
        
    def forward(self, x):
        # Encoder
        conv1 = self.activation(self.conv1_1(x))
        conv1 = self.activation(self.conv1_2(conv1))
        pool1 = self.pool1(conv1)
        pool1 = self.drop1(pool1)
        
        conv2 = self.activation(self.conv2_1(pool1))
        conv2 = self.activation(self.conv2_2(conv2))
        pool2 = self.pool2(conv2)
        pool2 = self.drop2(pool2)
        
        conv3 = self.activation(self.conv3_1(pool2))
        conv3 = self.activation(self.conv3_2(conv3))
        pool3 = self.pool3(conv3)
        pool3 = self.drop3(pool3)
        
        conv4 = self.activation(self.conv4_1(pool3))
        conv4 = self.activation(self.conv4_2(conv4))
        
        # Decoder with skip connections
        up_conv4 = self.upconv4(conv4)
        concat5 = torch.cat([up_conv4, conv3], dim=1)
        conv5 = self.activation(self.conv5_1(concat5))
        conv5 = self.activation(self.conv5_2(conv5))
        
        up_conv5 = self.upconv5(conv5)
        concat6 = torch.cat([up_conv5, conv2], dim=1)
        conv6 = self.activation(self.conv6_1(concat6))
        conv6 = self.activation(self.conv6_2(conv6))
        
        up_conv6 = self.upconv6(conv6)
        concat7 = torch.cat([up_conv6, conv1], dim=1)
        
        outputs = torch.concat([head(concat7) for head in self.heads], dim=1)        
        
        return outputs


class OptimizedUNet(nn.Module):
    def __init__(self, in_channels, out_channels, drop_prob=0.1, activation=torch.nn.ReLU):
        super(OptimizedUNet, self).__init__()
        self.opts = locals().copy()
        del self.opts['self']
        
        # Encoder - using strided convolutions instead of pooling
        self.conv1 = StridedConvBlock(in_channels, 32)
        self.conv2 = StridedConvBlock(32, 64)
        self.conv3 = StridedConvBlock(64, 128)
        self.conv4 = StridedConvBlock(128, 256)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DepthwiseSeparableConv2d(256, 512),
            nn.BatchNorm2d(512),
            activation(inplace=True)
        )
        
        # Decoder - using efficient upsampling
        self.up1 = EfficientUpsample(512, 256)
        self.up2 = EfficientUpsample(256, 128)
        self.up3 = EfficientUpsample(128, 64)
        self.up4 = EfficientUpsample(64, 32)
        
        # Heads
        self.heads = nn.ModuleList()
        for c in out_channels:
            self.heads.append(OptimizedHead(32, c))
        
        self.activation = activation(inplace=True)
        
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)  # 32 channels
        x2 = self.conv2(x1)  # 64 channels
        x3 = self.conv3(x2)  # 128 channels
        x4 = self.conv4(x3)  # 256 channels
        
        # Bottleneck
        x = self.bottleneck(x4)  # 512 channels
        
        # Decoder with skip connections
        x = self.up1(x, x4)  # 256 channels
        x = self.up2(x, x3)  # 128 channels
        x = self.up3(x, x2)  # 64 channels
        x = self.up4(x, x1)  # 32 channels
        
        # Heads
        outputs = torch.concat([head(x) for head in self.heads], dim=1)
        return outputs


if __name__ == "__main__":
    # Test BabyUNet with a sample input
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create a test input tensor of shape (1, 3, 128, 160)
    # Batch size 1, 3 input channels, height 128, width 160
    test_input = torch.randn(129, 3, 128, 160).to(device)
    
    # Create all UNet variants with 3 input channels and 1 output channel
    baby_unet = BabyUNet(in_channels=3, out_channels=[1]).to(device)
    medium_unet = MediumUNet(in_channels=3, out_channels=[1]).to(device)
    original_unet = UNet(in_channels=3, out_channels=[1]).to(device)
    optimized_unet = OptimizedUNet(in_channels=3, out_channels=[1]).to(device)
    
    # Forward pass
    output = baby_unet(test_input)
    
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Basic verification
    assert output.shape[0] == 129, "Batch size should remain 1"
    assert output.shape[1] == 1, "Output should have 1 channel as specified"
    assert output.shape[2:] == test_input.shape[2:], "Output spatial dimensions should match input"
    
    # Model size and complexity comparison
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Parameter count comparison
    baby_params = count_parameters(baby_unet)
    medium_params = count_parameters(medium_unet)
    original_params = count_parameters(original_unet)
    optimized_params = count_parameters(optimized_unet)
    
    print("\n--- Model Size Comparison ---")
    print(f"Original UNet parameters: {original_params:,}")
    print(f"MediumUNet parameters: {medium_params:,} ({medium_params/original_params*100:.1f}% of original)")
    print(f"BabyUNet parameters: {baby_params:,} ({baby_params/original_params*100:.1f}% of original)")
    print(f"OptimizedUNet parameters: {optimized_params:,} ({optimized_params/original_params*100:.1f}% of original)")
    
    # Inference time comparison
    import time
    
    def measure_inference_time(model, input_tensor, num_iterations=10000):  # Reduced iterations for testing
        # Warmup
        for _ in range(5):  # Reduced warmup iterations
            _ = model(input_tensor)
        
        # Measure time
        start_time = time.time()
        for _ in range(num_iterations):
            _ = model(input_tensor)
        end_time = time.time()
        
        return (end_time - start_time) / num_iterations
    
    print("\nMeasuring inference times (this may take a moment)...")
    baby_inference_time = measure_inference_time(baby_unet, test_input)
    medium_inference_time = measure_inference_time(medium_unet, test_input)
    original_inference_time = measure_inference_time(original_unet, test_input)
    optimized_inference_time = measure_inference_time(optimized_unet, test_input)
    
    print("\n--- Inference Speed Comparison ---")
    print(f"Original UNet inference time: {original_inference_time*1000:.2f} ms")
    print(f"MediumUNet inference time: {medium_inference_time*1000:.2f} ms ({medium_inference_time/original_inference_time*100:.1f}% of original)")
    print(f"BabyUNet inference time: {baby_inference_time*1000:.2f} ms ({baby_inference_time/original_inference_time*100:.1f}% of original)")
    print(f"OptimizedUNet inference time: {optimized_inference_time*1000:.2f} ms ({optimized_inference_time/original_inference_time*100:.1f}% of original)")
    
    print("\nTest passed successfully!")
