import os
import torch
import matplotlib.image as img
from torch.utils.data import Dataset
#from pathlib import Path


class EmotionDataset(Dataset):
	def __init__(self, data, path, transform=None):
		super().__init__
		self.data = data.values
		self.path = os.path.abspath(os.curdir) + path
		self.transform = transform
		self.y = data.emotion.values

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		_, img_name, label = self.data[index]
		img_path = os.path.join(self.path, img_name)
		image = img.imread(img_path)
		if self.transform is not None:
			image = self.transform(image)

		Y = self.y[index]

		sampler = {'image': torch.tensor(image),
				   'label': torch.tensor(Y, dtype=torch.long)}


		return sampler