from web_scraper import WebScraper
from data_preprocessing import DataPreprocessor
from data_annotation import DataAnnotator
from dataset_splitter import DatasetSplitter
from model_training import ModelTrainer
from model_testing import ModelTester
from deployment import ModelDeployer


# URLs for Kali Linux and msfconsole commands, tools, and documentation
kali_url = 'https://tools.kali.org/tools-listing'
msfconsole_url = 'https://www.offensive-security.com/metasploit-unleashed/msfconsole-commands/'

# Step 1: Scrape data from relevant websites
scraper = WebScraper(kali_url, msfconsole_url)
data = scraper.scrape_data()

# Step 2: Preprocess the collected data
preprocessor = DataPreprocessor()
processed_data = preprocessor.preprocess_data(data)

# Step 3: Assign tasks to each command in the preprocessed data
annotator = DataAnnotator()
annotated_data = annotator.annotate_data(processed_data)

# Step 4: Split the annotated data into training, validation, and testing sets
splitter = DatasetSplitter()
train_data, val_data, test_data = splitter.split_dataset(annotated_data)

# Step 5: Define and train a PyTorch model on the training dataset
trainer = ModelTrainer()
trainer.train_model(train_data, val_data)

# Step 6: Evaluate the performance of the trained model on the testing dataset
tester = ModelTester()
tester.test_model(test_data)

# Step 7: Deploy the trained model as an application that can automate Kali Linux tasks
deployer = ModelDeployer()
deployer.deploy_model(trainer.model)
