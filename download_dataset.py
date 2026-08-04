from torchvision.datasets import EuroSAT

dataset = EuroSAT(
    root="./dataset",
    download=True
)

print("Dataset downloaded successfully!")
print("Number of images:", len(dataset))