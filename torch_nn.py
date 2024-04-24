import torch
import PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data
train = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
dataset = DataLoader(train, batch_size=32, shuffle=True)
# 1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6), 10)
        )

    def forward(self, x):
        return self.model(x)
    
# Instance of the neural network, loss, optimizer
clf = ImageClassifier().to('mps')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# Training loop
if __name__ == '__main__':
    for epoch in range(10):
        for batch, (X, y) in enumerate(dataset):
            X, y = X.to('mps'), y.to('mps')
            y_pred = clf(X)
            loss = loss_fn(y_pred, y)

            # Apply backpropagation
            opt.zero_grad() # Zero existiong gradients
            loss.backward() # Compute gradients
            opt.step() # Take a step in the opposite direction of the gradient
        
        print(f"Epoch {epoch} Batch {batch+1} Loss {loss.item()}")
    
    with open('model_state.pt', 'wb') as f:
        save(clf.state_dict(), f)

    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))
    
    img = Image.open('test.png')
    img_tensor = ToTensor()(img).unsqueeze(0).to('mps')

    print(clf(img_tensor).argmax().item())