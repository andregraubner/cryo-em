from cryoet_data_portal import Client, Dataset
from tqdm import tqdm

client = Client()

dataset_ids = [
    10108, # cryo-FIB INS-1E
]
for dataset_id in dataset_ids:
    
    dataset = Dataset.get_by_id(client, dataset_id)
    for run in tqdm(dataset.runs):
        fname = f"data/dump/{dataset.id}_{run.name}.mrc"
        if not os.path.exists(fname):
            run.tomograms[0].download_mrcfile(fname)
        else:
            print("Already exists: skipping", fname)