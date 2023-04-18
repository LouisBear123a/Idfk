import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, train_data, val_data, vocab_size, embedding_dim, hidden_dim, output_dim, batch_size, lr, num_epochs):
        self.train_data = train_data
        self.val_data = val_data
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.model = self.build_model()
        self.train_loader, self.val_loader = self.build_loaders()

    def build_model(self):
        model = CommandClassifier(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim)
        return model

    def build_loaders(self):
        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_data, batch_size=self.batch_size)
        return train_loader, val_loader

    def evaluate_on_validation_set(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        with torch.no_grad():
            val_loss = 0.0
            val_acc = 0.0
            self.model.eval()
            for inputs, targets in self.val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_acc += (predicted == targets).sum().item()
            val_loss /= len(self.val_loader)
            val_acc /= len(self.val_data)
        return val_loss, val_acc

    def evaluate_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        train_losses = []
        val_losses = []
        val_accs = []
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            self.model.train()
            for inputs, targets in self.train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            
            val_loss, val_acc = self.evaluate_on_validation_set()
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f'Epoch {epoch + 1}/{self.num_epochs}:')
            print(f'Train Loss: {train_loss:.3f} | Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}')
        
        self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), 'command_classifier.pt')
