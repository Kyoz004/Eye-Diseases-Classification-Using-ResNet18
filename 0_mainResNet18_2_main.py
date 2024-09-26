# Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm.notebook import tqdm  # Thư viện giúp hiển thị thanh tiến trình
import torch
from torchsummary import summary
import torchmetrics
import warnings
import pathlib
from termcolor import colored  # Thư viện để in ra màu sắc trong console
from datetime import datetime
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
# Thư viện để phân chia dữ liệu thành các tập huấn luyện, validation và testing
import splitfolders  

torch.cuda.empty_cache() # Remove cache before run the cpu
warnings.filterwarnings('ignore') 
sns.set_style('darkgrid')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# Path to dataset
data_path = 'Eye-Diseases-Dataset'
# Create folders for train, validation, and test sets if they don't exist
output_path = 'imgs'
train_path = os.path.join(output_path, 'train')
val_path = os.path.join(output_path, 'val')
test_path = os.path.join(output_path, 'test')
if not os.path.exists(train_path) or not os.path.exists(val_path) or not os.path.exists(test_path):
    splitfolders.ratio(input=data_path, output=output_path, seed=42, ratio=(0.7, 0.15, 0.15))
else:
    print(colored("Data already split, skipping splitfolders...", "yellow"))

# Path of split folders
base_dir = pathlib.Path(output_path)

# Define transformation with augmentation
transform_train = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(10),  # Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

transform_test = transforms.Compose([
    transforms.Resize((250, 250)),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])

batch_size = 128

# Create Data Loaders with Augmentation for training set
train_ds = torchvision.datasets.ImageFolder(os.path.join(base_dir, 'train'), transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# Tạo Data Loaders cho tập validation
val_ds = torchvision.datasets.ImageFolder(os.path.join(base_dir, 'val'), transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=True)
# Tạo Data Loaders cho tập test
test_ds = torchvision.datasets.ImageFolder(os.path.join(base_dir, 'test'), transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

print(colored(f'Train Folder :\n ', 'green', attrs=['bold']))
print(train_ds)
print(colored(f'Validation Folder :\n ', 'green', attrs=['bold']))
print(val_ds)
print(colored(f'Test Folder :\n ', 'green', attrs=['bold']))
print(test_ds)

# Print shape of dataset for each set
for key, value in {'Train': train_loader, "Validation": val_loader, 'Test': test_loader}.items():
    for X, y in value:
        print(colored(f'{key}:', 'green', attrs=['bold']))
        print(f"Shape of images [Batch_size, Channels, Height, Width]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}\n")
        print('-'*45)
        break

# labels_map
labels_map = {i: class_name for i, class_name in enumerate(train_ds.classes)}
# Lấy thông tin về nhãn từ một batch dữ liệu huấn luyện
for imgs, labels in train_loader:
    break
print('Labels: ', labels)
print('-'*45)
num_classes = len(set(labels.numpy()))
print(f'Number of classes: {num_classes}')
# Hiển thị một batch ảnh từ data loader và nhãn tương ứng
plt.subplots(4, 8, figsize=(16, 8))
plt.suptitle('EyesDiseases in 1 Batch', fontsize=25, fontweight='bold')
for i in range(32):
    ax = plt.subplot(4, 8, i+1)
    img = torch.permute(imgs[i], (1, 2, 0))
    plt.imshow(img)
    label = labels_map[int(labels[i])]
    plt.title(label)
    plt.axis('off')
# Increase space between subplots
plt.subplots_adjust(hspace=0.5, wspace=1.0)
plt.savefig("SampleDiseases.png")
plt.show()

# CNN
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torchvision.models.resnet18(pretrained=True)
         # Đóng băng các tham số trừ 15 lớp cuối cùng
        for param in list(self.base.parameters())[:-15]:
            param.requires_grad = False
            
        self.block = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased dropout rate
            nn.Linear(128, num_classes),
        )
        self.base.fc = nn.Sequential()
        
    def get_optimizer(self):
        return torch.optim.SGD([
            {'params': self.base.parameters(), 'lr': 3e-5},
            {'params': self.block.parameters(), 'lr': 8e-4}
        ], momentum=0.9)
        
    def forward(self, x):
        x = self.base(x)
        x = self.block(x)
        return x

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# EarlyStopping để kiểm soát việc dừng sớm khi đạt tới mức accuracy cao nhất
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
# huấn luyện mô hình
class Trainer(nn.Module):
    def __init__(self, train_loader, val_loader, test_loader, device, early_stopping=None):
        super().__init__()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.model = Net().to(self.device)
        self.optimizer = self.model.get_optimizer()
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        self.loss_fxn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(self.device)
        self.early_stopping = early_stopping
        
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def training_step(self, x, y):
        self.model.train()
        pred = self.model(x)
        loss = self.loss_fxn(pred, y)
        acc = self.accuracy(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, acc
    
    def val_step(self, x, y):
        self.model.eval()
        with torch.no_grad():
            pred = self.model(x)
            loss = self.loss_fxn(pred, y)
            acc = self.accuracy(pred, y)
        return loss, acc
    
    def step_fxn(self, loader, step):
        loss, acc = 0, 0
        
        for X, y in tqdm(loader):
            X, y = X.to(self.device), y.to(self.device)
            l, a = step(X, y)
            loss, acc = loss + l.item(), acc + a.item()
        return loss / len(loader), acc / len(loader)
    
    def train(self, epochs):
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.step_fxn(self.train_loader, self.training_step)
            val_loss, val_acc = self.step_fxn(self.val_loader, self.val_step)
            
            for item, value in zip(self.history.keys(), list([train_loss, val_loss, train_acc, val_acc])):
                self.history[item].append(value)
            
            print("[Epoch: {}] Train: [loss: {:.3f} acc: {:.3f}] Val: [loss: {:.3f} acc:{:.3f}]".format(epoch + 1, train_loss, train_acc, val_loss, val_acc))
            
            self.scheduler.step(val_loss)

            # Early stopping
            if self.early_stopping:
                self.early_stopping(val_loss)
                if self.early_stopping.early_stop:
                    print("Early stopping")
                    break

            # Print the number of images in each set
            print(f"Number of training images: {len(self.train_loader.dataset)}")
            print(f"Number of validation images: {len(self.val_loader.dataset)}")
            print(f"Number of test images: {len(self.test_loader.dataset)}")

    def plot_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))
        ax[0].plot(self.history['train_acc'], label='Train Accuracy')
        ax[0].plot(self.history['val_acc'], label='Validation Accuracy')
        ax[0].set_title('Accuracy')
        ax[0].legend()

        ax[1].plot(self.history['train_loss'], label='Train Loss')
        ax[1].plot(self.history['val_loss'], label='Validation Loss')
        ax[1].set_title('Loss')
        ax[1].legend()
        plt.savefig('TrainingValidationWithSGD.png')
        plt.show()
        
    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            t0 = datetime.now()
            test_loss = []
            n_correct = 0
            n_total = 0

            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                y_pred = self.model(images)
                loss = self.loss_fxn(y_pred, labels)
                test_loss.append(loss.item())

                _, prediction = torch.max(y_pred, 1)
                n_correct += (prediction == labels).sum().item()
                n_total += labels.shape[0]

            test_loss = np.mean(test_loss)
            test_acc = n_correct / n_total
            dt = datetime.now() - t0
            print(colored(f'Test Loss: {test_loss:.4f}\nTest Accuracy: {test_acc:.4f}\nDuration: {dt}', 'green', attrs=['bold']))
            
            # Print the number of images in each set
            print(f"Number of training images: {len(self.train_loader.dataset)}")
            print(f"Number of validation images: {len(self.val_loader.dataset)}")
            print(f"Number of test images: {len(self.test_loader.dataset)}")

    def get_predictions(self):
        y_true = []
        y_pred = []
        correct_images = []
        incorrect_images = []
        self.model.eval()
        
        for images, labels in self.test_loader:
            images = images.to(self.device)
            labels = labels.numpy()
            outputs = self.model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(preds)
            
            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    correct_images.append(images[i].cpu())
                else:
                    incorrect_images.append((images[i].cpu(), labels[i], preds[i]))
                    
        return np.array(y_true), np.array(y_pred), correct_images, incorrect_images

epochs = 50
early_stopping = EarlyStopping(patience=5, min_delta=0.01)
trainer = Trainer(train_loader, val_loader, test_loader, device, early_stopping=early_stopping)
trainer.train(epochs)
trainer.plot_history()
trainer.evaluate()

# Generate predictions and classification report
y_true, y_pred, correct_images, incorrect_images = trainer.get_predictions()
class_names = list(train_ds.classes)

print(colored('Confusion Matrix:', 'green', attrs=['bold']))
print(confusion_matrix(y_true, y_pred))

print(colored('Classification Report:', 'green', attrs=['bold']))
print(classification_report(y_true, y_pred, target_names=class_names))

print(f'Number of correct predictions: {len(correct_images)}')
print(f'Number of incorrect predictions: {len(incorrect_images)}')

# Vẽ một lô hình ảnh từ bộ nạp thử nghiệm với nhãn thực và nhãn dự đoán
def plot_predictions(model, data_loader, labels_map):
    model.eval()
    correct_images = []
    incorrect_images = []
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            preds = preds.argmax(dim=1)

            plt.subplots(4, 8, figsize=(16, 12))
            plt.suptitle('Eye Diseases images in 1 Batch', fontsize=25, fontweight='bold')
            for i in range(32):
                ax = plt.subplot(4, 8, i+1)
                img = torch.permute(imgs[i].cpu(), (1, 2, 0))
                plt.imshow(img)
                true_label = labels_map[int(labels[i])]
                pred_label = labels_map[int(preds[i])]
                ax.set_title(f'True: {true_label}\nPred: {pred_label}', fontsize=8)
                plt.axis('off')

            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            plt.savefig('EyeDiseasesPredict.png')
            plt.show()

            # Lưu hình ảnh đúng và sai để vẽ biểu đồ sau
            for i in range(len(labels)):
                if preds[i] == labels[i]:
                    correct_images.append(imgs[i].cpu())
                else:
                    incorrect_images.append((imgs[i].cpu(), labels[i].cpu(), preds[i].cpu()))

            break

    return correct_images, incorrect_images

correct_images, incorrect_images = plot_predictions(trainer.model, test_loader, labels_map)

def plot_prediction_results(correct_images, incorrect_images, labels_map):
    # Plot correct predictions
    plt.subplots(4, 8, figsize=(16, 12))
    plt.suptitle('Correct Predictions', fontsize=25, fontweight='bold')
    for i in range(min(32, len(correct_images))):
        ax = plt.subplot(4, 8, i+1)
        img = torch.permute(correct_images[i], (1, 2, 0))
        plt.imshow(img)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig('CorrectPredictions.png')
    plt.show()

    # Plot incorrect predictions
    plt.subplots(4, 8, figsize=(16, 12))
    plt.suptitle('Incorrect Predictions', fontsize=25, fontweight='bold')
    for i in range(min(32, len(incorrect_images))):
        ax = plt.subplot(4, 8, i+1)
        img, true_label, pred_label = incorrect_images[i]
        img = torch.permute(img, (1, 2, 0))
        plt.imshow(img)
        true_label_name = labels_map[int(true_label)]
        pred_label_name = labels_map[int(pred_label)]
        ax.set_title(f'True: {true_label_name}\nPred: {pred_label_name}', fontsize=8)
        plt.axis('off')
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig('IncorrectPredictions.png')
    plt.show()

plot_prediction_results(correct_images, incorrect_images, labels_map)

def plot_confusion_matrix(trainer, loader):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(trainer.device), y.to(trainer.device)
            pred = trainer.model(X)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(pred.argmax(dim=1).cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels_map.values(), yticklabels=labels_map.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('ConfusionMatrix.png')
    plt.show()

# Tạo instance của Trainer và train
trainer = Trainer(train_loader, val_loader, test_loader, device, early_stopping=EarlyStopping(patience=7, min_delta=0.01))
# Plot the confusion matrix for the test set
plot_confusion_matrix(trainer, test_loader)