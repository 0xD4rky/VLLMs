import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

model = models.resnet50(pretrained=True)

model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet normalization
])

image_path = "/Users/darky/Documents/dyna_vec/vectorization/img/test.jpg" 
image = Image.open(image_path).convert('RGB')
input_tensor = transform(image).unsqueeze(0)
with torch.no_grad():
    embeddings = model(input_tensor).squeeze()

model = torch.nn.Sequential(
    torch.nn.Linear(2048,300)
)
embeddings_i = model(embeddings)