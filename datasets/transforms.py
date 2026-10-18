"""Image preprocessing and data augmentation."""
from PIL import Image
from torchvision import transforms

def square_generated(image):
    w, h = image.size
    output = Image.new(image.mode, (max(w, h), max(w, h)), color=0)
    point = abs(w-h)//2
    output.paste(image, (point, 0) if w<h else (0, point))
    return output

def expand_to_3_channels(image):
    return image.repeat(3, 1, 1) if image.size(0) != 3 else image

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(expand_to_3_channels),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Lambda(expand_to_3_channels),
    ])
