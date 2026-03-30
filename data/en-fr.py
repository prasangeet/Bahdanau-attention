import kagglehub
from kagglehub import KaggleDatasetAdapter
import os

# Save the downloaded dataset into a local folder that the project can reuse.
file_path = "dataset"
file_name = "english_french.csv"

os.makedirs(file_path, exist_ok=True)

# Download the CSV as a pandas DataFrame through KaggleHub.
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "rajpulapakura/english-to-french-small-dataset",
  file_name,
  # Extra loader options can be passed here if the dataset access needs them.
  # See the KaggleHub README for supported adapter arguments.
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

print("First 5 records: ", df.head())

# Persist the dataset so training code can load it from disk later.
save_path = os.path.join(file_path, file_name)
df.to_csv(save_path, index=False)

print(f"\nDataset Saved: {save_path}")
