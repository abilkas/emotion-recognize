import torch
import torch.nn as nn
import torchvision
from data_aug import legend
from transforms import transform

random_seed = 47
torch.manual_seed(random_seed);

label_path = 'data/legend.csv'
image_path = '/data/images/'

data_legend = legend(label_path)
num_epoch = 15
num_classes = len(data_legend.emotion.unique())
batch_size = 20
learning_rate = 0.0001

#CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torchvision.models.resnet18(pretrained=True)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

model.conv1.in_channels = 1

train_loader, valid_loader = transform(label_path, image_path, batch_size)

train_losses = []
valid_losses = []

print('---train---')
for epoch in range(num_epoch):
	train_loss = 0.0
	valid_loss = 0.0

	model.train()
	for datas in train_loader:
		image, target = datas['image'].cuda(), datas['label'].cuda()


		optimizer.zero_grad()

		output = model(image)

		loss = criterion(output, target)

		loss.backward()

		optimizer.step()

		train_loss = train_loss + (loss.item() * image.size(0))

	model.eval()
	for datas in valid_loader:
		image, target = datas['image'].cuda(), datas['label'].cuda()

		output = model(image)

		loss = criterion(output, target)

		valid_loss = valid_loss + (loss.item() * image.size(0))

	train_loss = train_loss/len(train_loader.sampler)
	valid_loss = valid_loss/len(valid_loader.sampler)
	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	print('Epoch: {} \tTraining loss: {:.6f} \tValid loss: {:.6f}'.format(epoch, train_loss, valid_loss))