from web_scraper import WebScraper
from data_preprocessing import DataPreprocessing
from data_annotation import DataAnnotation
from dataset_splitter import DatasetSplitter
from model_training import ModelTrainer
from model_testing import ModelTester
from deployment import CommandPredictor

# Step 1: Collect data from relevant websites using WebScraper
kali_url = 'https://tools.kali.org/tools-listing'
msfconsole_url = 'https://www.hackingarticles.in/comprehensive-guide-to-metasploit-msfconsole/'
scraper = WebScraper(kali_url, msfconsole_url)
data = scraper.scrape_data()

# Step 2: Preprocess the data using DataPreprocessing
preprocessor = DataPreprocessing()
data = preprocessor.preprocess_data(data)

# Step 3: Assign tasks to each command in the preprocessed data using DataAnnotation
annotator = DataAnnotation()
annotated_data = annotator.annotate_data(data)

# Step 4: Split the annotated data into training, validation, and testing sets using DatasetSplitter
splitter = DatasetSplitter()
train_data, val_data, test_data = splitter.split_data(annotated_data)

# Step 5: Train a PyTorch model on the training set and evaluate its performance on the validation set using ModelTrainer
vocab_size = preprocessor.vocab_size
embedding_dim = 100
hidden_dim = 256
output_dim = annotator.num_tasks
batch_size = 64
lr = 0.001
num_epochs = 20
trainer = ModelTrainer(train_data, val_data, vocab_size, embedding_dim, hidden_dim, output_dim, batch_size, lr, num_epochs)
trainer.train_model()

# Step 6: Evaluate the performance of the trained model on the testing set using ModelTester
tester = ModelTester(test_data, vocab_size, embedding_dim, hidden_dim, output_dim)
tester.test_model()

# Step 7: Deploy the trained model as an application that can automate Kali Linux tasks using CommandPredictor
predictor = CommandPredictor('command_classifier.pt')

# Example usage of CommandPredictor to predict tasks for new commands
command1 = 'nmap -sS 192.168.0.1'
task1 = predictor.predict(command1)
print(f'Task for command "{command1}": {task1}')

command2 = 'msfconsole'
task2 = predictor.predict(command2)
print(f'Task for command "{command2}": {task2}')
