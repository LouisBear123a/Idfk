import pandas as pd
from sklearn.model_selection import train_test_split

class DatasetSplitter:
    def __init__(self):
        pass
    
    def split_data(self, data, test_size=0.2):
        # Split the annotated data into training, validation, and testing sets
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=test_size, random_state=42)
        return train_data, val_data, test_data
    
    def convert_to_dataframe(self, data):
        # Convert the data to a Pandas DataFrame
        df = pd.DataFrame(columns=['command', 'task'])
        for item in data:
            for command in item:
                df = df.append({'command': command['command'], 'task': command['task']}, ignore_index=True)
        return df
