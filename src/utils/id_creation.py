import hashlib
import os
import shutil

def dataset_id_from_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def save_uploaded_dataset(uploaded_path):
    """
    Takes the path of a user-uploaded CSV file and renames it to:
        data/raw/<dataset_id>.csv
    
    Returns:
        new_path (str): The normalized, renamed dataset path.
        dataset_id (str): Unique dataset identifier.
    """
    dataset_id = dataset_id_from_file(uploaded_path)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_dir = os.path.join(base_dir, "data", "raw")

    os.makedirs(raw_dir, exist_ok=True)

    new_path = os.path.join(raw_dir, f"{dataset_id}.csv")

    if not os.path.exists(new_path):
        shutil.move(uploaded_path, new_path)
        print(f"Dataset saved as: {new_path}")
    else:
        print(f"File already exists for dataset_id={dataset_id}, reusing existing file.")

    return new_path, dataset_id
