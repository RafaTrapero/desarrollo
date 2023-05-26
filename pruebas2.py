import deeplake
from torchvision import datasets, transforms, models

# DeepLake
ds = deeplake.load('hub://activeloop/liar-train')



# Pytorch
dataloader = ds.pytorch(num_workers=0, batch_size=4, shuffle=False)

