import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F
import timm
from sklearn.metrics import mean_absolute_error, r2_score
#import scikitplot as skplt
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_pred, model_name, labels):
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {model_name}')
    plt.legend(loc="lower right")
    plt.savefig(f'{model_name}_roc_curve.png')
    plt.close()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("----->",device)

# --- 1. Data Loading and Preprocessing ---

# Load the dataset
df = pd.read_csv('archive/Data_Entry_2017.csv')

# Create a list of all image paths
image_paths = glob('archive/images*/images/*.png')

# Create a new column for the full image path
df['path'] = df['Image Index'].map({os.path.basename(x): x for x in image_paths})

# Get all unique labels
all_labels = np.unique(np.concatenate([l.split('|') for l in df['Finding Labels']]))
all_labels = [l for l in all_labels if l != 'No Finding']

# One-hot encode the labels and ensure they are numeric
for label in all_labels:
    df[label] = df['Finding Labels'].apply(lambda x: 1 if label in x else 0).astype('float32')

# --- 2. Splitting the Data ---

# Split the data into training (70%) and the rest (30%)
train_df, rest_df = train_test_split(df, test_size=0.3, random_state=42)

# Split the rest (30%) into validation (15%) and testing (15%)
val_df, test_df = train_test_split(rest_df, test_size=0.5, random_state=42)

# --- 3. PyTorch Dataset and DataLoaders ---

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, labels, transform=None):
        self.dataframe = dataframe
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['path']
        image = Image.open(img_path).convert('RGB')
        label_values = self.dataframe.iloc[idx][self.labels].to_numpy(dtype='float32')
        label = torch.from_numpy(label_values)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create datasets
train_dataset = ChestXrayDataset(train_df, all_labels, transform=data_transforms['train'])
val_dataset = ChestXrayDataset(val_df, all_labels, transform=data_transforms['val'])

# Create dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# --- 4. Model Definition ---
def get_model(model_name, num_labels, pretrained=True):
    model = None
    if model_name == 'DenseNet-121':
        model = models.densenet121(pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )
    elif model_name == 'ViT':
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        num_ftrs = model.head.in_features
        model.head = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )
    elif model_name == 'ResNet':
        model = models.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )
    elif model_name == 'EfficientNet':
        model = timm.create_model('efficientnet_b0', pretrained=pretrained)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
        )
    
    return model

models_to_test = ['DenseNet-121', 'ViT', 'ResNet', 'EfficientNet']
results = []

# --- 5. Loss Function and Optimizer ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

criterion = FocalLoss()

# --- 6. Training and Validation ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=10, return_preds=False):
    best_auc_roc = 0.0
    best_model_wts = model.state_dict()
    
    final_preds = None
    final_labels = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            if i % 200 == 199:
                print(f'  [Batch {i + 1}/{len(train_loader)}] Training Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_dataset)
        print(f'Training Loss: {epoch_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                preds = torch.sigmoid(outputs)
                val_preds.append(preds.cpu().numpy())
                val_labels.append(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels)
        auc_roc = roc_auc_score(val_labels, val_preds, average='macro')
        print(f'Validation Loss: {val_loss:.4f}, Validation AUC-ROC: {auc_roc:.4f}')

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_model_wts = model.state_dict()
            final_preds = val_preds
            final_labels = val_labels
            print(f'New best model saved with AUC-ROC: {best_auc_roc:.4f}')

    model.load_state_dict(best_model_wts)
    
    if return_preds:
        return model, final_preds, final_labels
    return model

# --- 7. Benchmarking Loop ---
for model_name in models_to_test:
    print(f"\n--- Training and evaluating {model_name} ---")
    
    # Create model
    model = get_model(model_name, len(all_labels))
    model = model.to(device)

    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze the new classifier
    if hasattr(model, 'classifier'):
        for param in model.classifier.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    elif hasattr(model, 'head'):
        for param in model.head.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.head.parameters(), lr=0.001)
    elif hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    # Train the classifier head
    print("Training the classifier head...")
    model = train_model(model, criterion, optimizer, scheduler, num_epochs=10)

    # Unfreeze all layers for fine-tuning
    print("\nUnfreezing all layers for fine-tuning...")
    for param in model.parameters():
        param.requires_grad = True

    optimizer_ft = optim.Adam(model.parameters(), lr=0.00001)
    scheduler_ft = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=1, verbose=True)

    # Fine-tune the entire model
    print("Fine-tuning the entire model...")
    model, val_preds, val_labels = train_model(model, criterion, optimizer_ft, scheduler_ft, num_epochs=10, return_preds=True)

    # Calculate metrics
    val_preds_binary = (val_preds > 0.5).astype(int)
    
    auc_roc = roc_auc_score(val_labels, val_preds, average='macro')
    mae = mean_absolute_error(val_labels, val_preds)
    r2 = r2_score(val_labels, val_preds)
    accuracy = accuracy_score(val_labels, val_preds_binary)
    f1 = f1_score(val_labels, val_preds_binary, average='macro')

    results.append({
        'Model': model_name,
        'AUC-ROC': auc_roc,
        'MAE': mae,
        'R2': r2,
        'Accuracy': accuracy,
        'F1-score': f1
    })

    # Plot ROC curve
    plot_roc_curve(val_labels, val_preds, model_name, all_labels)
    print(f"ROC curve plot saved to {model_name}_roc_curve.png")

    # Save the final model
    model_save_path = f'{model_name}_chest_xray_model.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

# --- 8. Results ---
results_df = pd.DataFrame(results)
print("\n--- Benchmarking Results ---")
print(results_df)
results_df.to_csv('benchmarking_results.csv', index=False)
print("Results saved to benchmarking_results.csv")
