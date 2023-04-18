import torch

class CommandPredictor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = self.load_model()
        self.task_dict = {'task1': ['command1', 'command2'], 'task2': ['command3', 'command4']}
    
    def load_model(self):
        # Load the trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = CommandClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)
        model.load_state_dict(torch.load(self.model_path))
        model.eval()
        return model
    
    def predict(self, command):
        # Predict the task associated with the given command
        with torch.no_grad():
            input_tensor = torch.tensor([self.get_command_index(command)], dtype=torch.long).unsqueeze(0)
            output_tensor = self.model(input_tensor)
            output_index = torch.argmax(output_tensor).item()
            task = self.get_task(output_index)
            return task
    
    def get_command_index(self, command):
        # Convert the command to its corresponding index in the vocabulary
        vocab = {'command1': 0, 'command2': 1, 'command3': 2, 'command4': 3}
        return vocab.get(command, len(vocab))
    
    def get_task(self, index):
        # Convert the output index to its corresponding task
        for task, commands in self.task_dict.items():
            if index in [self.get_command_index(command) for command in commands]:
                return task
        return 'unknown'
