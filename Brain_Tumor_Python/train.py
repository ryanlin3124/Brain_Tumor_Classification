import os 
import torch
from torchvision import transforms

import custom_model, load_data, train_test, utils

EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.001

train_dir = "C:\Ryan Lin Files\Brain_Tumor_Python\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Train"
test_dir = "C:\Ryan Lin Files\Brain_Tumor_Python\Epic and CSCR hospital Dataset\Epic and CSCR hospital Dataset\Test"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transfroms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
])

train_dataloader, test_dataloader, class_name = load_data.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transforms=data_transfroms,
    batch_size=BATCH_SIZE
)

model = custom_model.Model_Custom(
    input_shape=3,
    output_shape=len(class_name)
).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

train_test.tr