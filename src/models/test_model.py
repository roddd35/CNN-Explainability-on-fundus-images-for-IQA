import torch
import torchvision
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from train_model import Net
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, precision_score, recall_score

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

'''
Run the model on the test_loader
images and calculate its metrics.
Only pytorch needed for this step!
'''
def evaluate_model(model, test_loader, loss_fn):
	model.eval()

	running_loss = 0.0
	accuracy = 0
	total = 0

	y_true = []
	y_pred = []
	y_pred_prob = []

	with torch.no_grad():
		for i, data in enumerate(test_loader):
			images, labels = data
			images, labels = images.to(device), labels.to(device)

			outputs = model(images)

			loss = loss_fn(outputs, labels)
			running_loss += loss.item()

			probabilities = torch.softmax(outputs, dim=1)

			_, predicted = torch.max(outputs.data, 1)

			total += labels.size(0)
			accuracy += (predicted == labels).sum().item()

			y_true.extend(labels.cpu().numpy())
			y_pred.extend(predicted.cpu().numpy())
			y_pred_prob.extend(probabilities.cpu().numpy())

	avg_loss = running_loss / (i + 1)
	accuracy = 100 * accuracy / total

	return avg_loss, accuracy, y_true, y_pred, np.array(y_pred_prob)

'''
Print all the metrics values
'''
def display_metrics(loss, accuracy, y_true, y_pred):
	target_names = ['Inadequate', 'Adequate']

	print(f"\nModel metrics results:\n")
	print(f"{'-'*45}")

	print(f"Accuracy: {accuracy}")
	print(f"Loss: {loss}")
	print(f"Precision: {precision_score(y_true, y_pred, average='weighted')}")
	print(f"Recall: {recall_score(y_true, y_pred, average='weighted')}")
	print(f"AUC (ROC): {roc_auc_score(y_true, y_pred, average='weighted')}")

	print('Confusion Matrix')
	cm = confusion_matrix(y_true, y_pred)
	print(cm)

	print('Classification Report')
	print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

	return cm


'''
Handle all the test algorithms
for the model on the test_loader.
'''
def test_model(test_loader, path):
	loss_fn = nn.CrossEntropyLoss()

	model = Net()
	model = model.to(device)
	model.load_state_dict(torch.load(path, weights_only=True))

	loss, accuracy, y_true, y_pred, y_pred_prob = evaluate_model(model, test_loader, loss_fn)
	cm = display_metrics(loss, accuracy, y_true, y_pred)

	return y_true, y_pred, y_pred_prob, cm