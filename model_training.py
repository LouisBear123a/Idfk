import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score

class CommandDataset(Dataset):
    def __init__(self, commands, tasks):
        self.commands = commands
        self.tasks = tasks
    
    def __len__(self):
        return len(self.commands)
    
    def __getitem__(self, idx):
        return self.commands[idx], self.tasks[idx]

class CommandClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        predictions = self.fc(hidden[-1])
        return predictions.squeeze(1)

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
    
    def train_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define the model, optimizer, and loss function
        model = CommandClassifier(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss().to(device)
        
        # Create data loaders
        train_dataset = CommandDataset(self.train_data['commands'], self.train_data['tasks'])
        val_dataset = CommandDataset(self.val_data['commands'], self.val_data['tasks'])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Train the model
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc = self.evaluate_model(model, val_loader, criterion, device)
            print(f'Epoch {epoch+1}/{self.num_epochs}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Val Loss: {val_loss:.3f}, Val Acc: {val_acc:.3f}')
        
        # Save the trained model
        torch.save(model.state_dict(), 'command_classifier.pt')
    
    def train_epoch(self, model, data_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0
        total_correct = 0
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total_correct += self.get_num_correct(outputs, targets)
        return total_loss / len(data_loader.dataset), total_correct / len(data_loader.dataset)
    
    def evaluate_model(self
