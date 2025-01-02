from cryoet_data_portal import Client, Dataset

client = Client()

dataset = Dataset.get_by_id(client, 10441)
dataset.download_everything("./data")