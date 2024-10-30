import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# tentar trocar o adam aqui e no pytorch e treinar novamente
# usar SGD ou RMSPROP

'''
Define the CNN architecture
based on ResNet50, with its
weights frozen.
Train only the new fc-layer
'''
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
		self.features = self.model.features	# self.features has only convolutional layers from self.model
		# self.model = nn.Sequential(*list(self.model.children())[:-1]) # remove fc layer
		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(25088, 512)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(512, 2)
		self.dropout = nn.Dropout(p=0.5)

	# x represents our data
	def forward(self, x):
		x = self.features(x)
		x = self.flatten(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)
		x = self.fc2(x)

		return x

'''
Define path and
save model to file
'''
def save_model(model):
	model_path = '/home/rodrigocm/research/gradcam-on-eye-fundus-images-IQA/data/model/brset_model.pth'
	torch.save(model.state_dict(), model_path)

'''
Test model on validation set
each epoch. Returns the model
accuracy and loss values
'''
def validate_model(model, val_loader, loss_fn):
	model.eval()
	accuracy = 0.
	running_loss = 0.
	total = 0.

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

	with torch.no_grad():
		for i, data in enumerate(val_loader):
			images, labels = data
			images, labels = images.to(device), labels.to(device)

			# run the model on the test set to predict labels
			outputs = model(images)

			# compute loss
			loss = loss_fn(outputs, labels)
			running_loss += loss.item()

			# use softmax to get probabilities
			probabilities = torch.softmax(outputs, dim=1)

			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			accuracy += (predicted == labels).sum().item()

	# compute the accuracy over all test images
	avg_loss = running_loss / (i + 1)
	accuracy = (100 * accuracy / total)

	return avg_loss, accuracy 

'''
Compute the training process for an epoch
updates the loss function and backpropagate
the model fully connected
'''
def train_one_epoch(model, train_loader, loss_fn, optimizer):
	running_loss = 0.
	running_acc = 0.
	total_batches = 0
	total = 0.

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	model.train(True)

	# iterate through the train_loader batches
	for i, data in enumerate(train_loader):
		images, labels = data
		images, labels = images.to(device), labels.to(device)

		# zero gradients for every batch
		optimizer.zero_grad()

		# make predictions for this batch (outputs are logits)
		outputs = model(images)

		# compute the loss and its gradients, call backgpropagation
		loss = loss_fn(outputs, labels)
		loss.backward()

		# adjust learning weights
		optimizer.step()

		# gather data and report
		running_loss += loss.item()
		total_batches += 1

		# calculate accuracy for this batch
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		running_acc += (predicted == labels).sum().item()

		# if i % 2 == 1:
		# 	print(f'->Batch {i+1} loss: {(running_loss / 2):.4f}')

	# compute overall accuracy for the epoch
	accuracy = 100 * running_acc / total
	avg_loss = running_loss / total_batches

	return avg_loss, accuracy

'''
Define training parameters and 
execute the model training process
Evaluate model according to validation
dataset.
'''
def train_model(train_loader, val_loader, EPOCHS, lr):
	model = Net()
	print(model)

	# define history arrays
	acc_hist = []
	vacc_hist = []
	loss_hist = []
	vloss_hist = []

	# freeze all layers in the ResNet50 backbone
	for param in model.features.parameters():
		param.requires_grad = False  # freeze all convolutional layers
	
	# defining parameters
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
	loss_fn = nn.CrossEntropyLoss()

	curr_epoch = 0
	best_acc = 0.

	# training step
	for epoch in range(EPOCHS):
		print(f'\n{"="*30}\nEPOCH {curr_epoch + 1}/{EPOCHS}\n{"="*30}')

		# run a training epoch
		model.train()
		avg_loss, accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer)

		# get validation results
		avg_vloss, vaccuracy = validate_model(model, val_loader, loss_fn)

		# formatted output
		print(f"\nResults after EPOCH {curr_epoch + 1}:\n")
		print(f"{'Metric':<15}{'Train':<15}{'Validation':<15}")
		print(f"{'-'*45}")
		print(f"{'Loss':<15}{avg_loss:.4f}{'':<7}{avg_vloss:.4f}")
		print(f"{'Accuracy':<15}{accuracy:.3f}{'':<7}{vaccuracy:.3f}")

		# Track best performance, and save the model's state
		if vaccuracy > best_acc:
			best_acc = vaccuracy
			save_model(model)
		
		# fill history arrays
		loss_hist.append(avg_loss)
		vloss_hist.append(avg_vloss)
		acc_hist.append(accuracy)
		vacc_hist.append(vaccuracy)

		if epoch == 10:
			optimizer = torch.optim.Adam(model.parameters(), lr=lr/10)
			for param in model.features[20:].parameters():	# unfreeze last 2 conv blocks (last 10 layers)
				param.requires_grad = True

		curr_epoch += 1
	return loss_hist, vloss_hist, acc_hist, vacc_hist