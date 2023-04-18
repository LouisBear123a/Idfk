class DataAnnotation:
    def __init__(self):
        pass
    
    def annotate_data(self, data):
        # Assign tasks to each command in the preprocessed data
        annotated_data = []
        for item in data:
            annotated_commands = []
            for command, task in zip(item['commands'], item['tasks']):
                annotated_commands.append({'command': command, 'task': task})
            annotated_data.append(annotated_commands)
        return annotated_data
