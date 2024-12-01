import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load a pretrained ResNet model (e.g., ResNet-50)
model = models.resnet50(pretrained=True)

# Remove the classification head (fc layer) to get embeddings
model = torch.nn.Sequential(*list(model.children())[:-1])

# Set the model to evaluation mode
model.eval()

# Define image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet expects 224x224 input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet normalization
])

# Load an image and apply transformations
image_path = "path_to_your_image.jpg"  # Replace with your image path
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Generate embeddings
with torch.no_grad():
    embeddings = model(input_tensor).squeeze().numpy()

print("Generated Embeddings Shape:", embeddings.shape)
print("Embeddings:", embeddings)
