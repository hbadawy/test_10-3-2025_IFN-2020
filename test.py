
import torch
import torch.nn.functional as F

# Example image tensor of shape (batch_size, channels, height, width)
image = torch.randn(1, 3, 256, 256)  # (batch_size=1, channels=3, height=256, width=256)

# Reshape the image to (height/2, width/2)
resized_image = F.interpolate(image, scale_factor=(1/2), mode='bilinear', align_corners=False)
resized_image2 = F.interpolate(image, scale_factor=(1/4), mode='bilinear', align_corners=False)
resized_image3 = F.interpolate(image, scale_factor=(1/8), mode='bilinear', align_corners=False)
resized_image4 = F.interpolate(image, scale_factor=(1/16), mode='bilinear', align_corners=False)

print(f"Original shape: {image.shape}")
print(f"Resized shape: {resized_image.shape, resized_image2.shape, resized_image3.shape, resized_image4.shape}")


