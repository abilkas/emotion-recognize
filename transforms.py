import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from datasets import EmotionDataset
from data_aug import legend

train_transform = transforms.Compose([transforms.ToPILImage(),
									  transforms.RandomHorizontalFlip(),
									  transforms.RandomRotation(20, resample=Image.BILINEAR),
									  transforms.Grayscale(num_output_channels=3),
									  transforms.Resize((224, 224)),
									  transforms.ToTensor(),
									  transforms.Normalize([0.5], [0.5])])

test_transform = transforms.Compose([transforms.ToPILImage(),
									 transforms.Grayscale(num_output_channels=3),
									 transforms.Resize((224, 224)),
									 transforms.ToTensor(),
									 transforms.Normalize([0.5], [0.5])])

valid_transform = transforms.Compose([transforms.ToPILImage(),
									  transforms.Grayscale(num_output_channels=3),
									  transforms.Resize((224, 224)),
									 transforms.ToTensor(),
									 transforms.Normalize([0.5], [0.5])])

	
def transform(label_path, image_path, batch_size):
	data_legend = legend(label_path)

	train, valid = train_test_split(data_legend, test_size=0.2, shuffle=True, stratify=data_legend.emotion)

	train_data = EmotionDataset(train, image_path, train_transform)
	valid_data = EmotionDataset(valid, image_path, valid_transform)

	train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
	valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False, num_workers=0)

	return train_loader, valid_loader