import torch
from sklearn.metrics import accuracy_score

class ModelTester:
    def __init__(self, test_data, vocab_size, embedding_dim, hidden_dim, output_dim):
        self.test_data = test_data
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def test_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the trained model
        model = CommandClassifier(self.vocab_size, self.embedding_dim, self.hidden_dim, self.output_dim).to(device)
        model.load_state_dict(torch.load('command_classifier.pt'))
        model.eval()
        
        # Create a data loader for the test set
        test_dataset = CommandDataset(self.test_data['commands'], self.test_data['tasks'])
        test_loader = DataLoader(test_dataset, batch_size=1)
        
        # Test the model on the test set
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                true_label = targets.cpu().item()
                predicted_label = torch.argmax(outputs).cpu().item()
                true_labels.append(true_label)
                predicted_labels.append(predicted_label)
        test_acc = accuracy_score(true_labels, predicted_labels)
        print(f'Test Acc: {test_acc:.3f}')
