import torch

class DataPreprocessing:
    def __init__(self):
        pass
    
    def tokenize_data(self, data):
        # Tokenize the text data into commands and tasks
        tokens = []
        for item in data:
            commands = item.split(' ')
            tasks = []
            for command in commands:
                tasks.append(self.get_task(command))
            tokens.append({'commands': commands, 'tasks': tasks})
        return tokens
    
    def get_task(self, command):
        # Determine the task associated with each command
        tasks = {'task1': ['command1', 'command2'], 'task2': ['command3', 'command4']}
        for task, commands in tasks.items():
            if command in commands:
                return task
        return 'unknown'
    
    def remove_irrelevant_data(self, data):
        # Remove irrelevant or redundant information
        processed_data = []
        for item in data:
            if item['tasks'][0] != 'unknown':
                processed_data.append(item)
        return processed_data
    
    def convert_to_tensors(self, data):
        # Convert the data into a format suitable for model training
        commands = []
        tasks = []
        for item in data:
            commands.append(item['commands'])
            tasks.append(item['tasks'])
        commands_tensor = torch.tensor(commands, dtype=torch.long)
        tasks_tensor = torch.tensor(tasks, dtype=torch.long)
        return commands_tensor, tasks_tensor
