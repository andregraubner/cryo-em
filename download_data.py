from cryoet_data_portal import Client, Dataset
from tqdm import tqdm
import os

client = Client()

dataset_ids = [
    10108, # cryo-FIB INS-1E
    10169, 
    10170
]
for dataset_id in dataset_ids:
    
    dataset = Dataset.get_by_id(client, dataset_id)
    for run in tqdm(dataset.runs):

        # download_mrcfile saves this *into* the folder specified by the path
        save_path = f"data/dump/{dataset.id}"
        fname = f"{save_path}/{run.name}.mrc"
        if not os.path.exists(fname):
            run.tomograms[0].download_mrcfile(save_path)
        else:
            print("Already exists: skipping", fname)