import torch
from PIL import Image
import matplotlib.pyplot as plt
from train import ConditionalUNet, denoise_image_with_condition, transform

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConditionalUNet().to(device)
model.load_state_dict(torch.load('sr.pth', map_location=device))
model.eval()


def load_custom_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image.to(device)

custom_lr_image_path = r'path/to/your/LR/image'  
custom_lr_image = load_custom_image(custom_lr_image_path, transform)
with torch.no_grad():
    denoised_image = denoise_image_with_condition(model, custom_lr_image)


# Convert tensors to numpy arrays for visualization
denoised_image_np = denoised_image.squeeze().cpu().numpy().transpose(1, 2, 0)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow((custom_lr_image.squeeze().cpu().numpy().transpose(1, 2, 0) * 0.5 + 0.5).clip(0, 1))
plt.title('LR Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow((denoised_image_np * 0.5 + 0.5).clip(0, 1)) # Denormalizing the image
plt.title('Output Image')
plt.axis('off')

plt.show()