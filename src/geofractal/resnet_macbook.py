import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm
import time
import os

os.environ['PYTORCH_MPS_FALLBACK_WARNING'] = '1'


def preload_to_device(loader, device):
    """Load entire dataset into MPS memory"""
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs).to(device), torch.cat(ys).to(device)


def main():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1024, shuffle=False, num_workers=0)

    print("Preloading dataset to MPS...")
    X_train, y_train = preload_to_device(trainloader, device)
    print(f"Dataset loaded: {X_train.shape} ({X_train.element_size() * X_train.nelement() / 1e6:.1f} MB)")

    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    batch_size = 512
    num_samples = X_train.size(0)
    num_batches = num_samples // batch_size

    print(f"\nTraining ResNet18 on CIFAR-10 | Device: {device}")
    print(f"Batch size: {batch_size} | Batches: {num_batches}")
    print("-" * 50)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        perm = torch.randperm(num_samples, device=device)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        torch.mps.synchronize()
        start = time.time()

        pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1:2d}", leave=True)
        for i in pbar:
            inputs = X_shuffled[i*batch_size:(i+1)*batch_size]
            labels = y_shuffled[i*batch_size:(i+1)*batch_size]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{100*correct/total:.2f}%"})

        scheduler.step()

        torch.mps.synchronize()
        elapsed = time.time() - start
        throughput = num_samples / elapsed

        print(f"         Loss: {running_loss/num_batches:.4f} | "
              f"Acc: {100*correct/total:.2f}% | {elapsed:.1f}s | {throughput:.0f} img/s")

    print("-" * 50)
    print("Done!")


if __name__ == '__main__':
    main()