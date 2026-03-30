import torch
from torch.utils.data import Dataset, DataLoader

class TranslationDataset(Dataset):
    """Expose paired English and French token sequences as PyTorch samples."""

    def __init__(self, eng_data, fre_data) -> None:
        """Store the already-tokenized parallel corpus."""
        self.eng_data = eng_data
        self.fre_data = fre_data

    def __len__(self):
        """Return the number of aligned sentence pairs available."""
        return len(self.eng_data)

    def __getitem__(self, index):
        """Fetch one sentence pair and cast both sequences to tensors."""
        return (
            torch.tensor(self.eng_data[index], dtype=torch.long),
            torch.tensor(self.fre_data[index], dtype=torch.long)
        )

class DataLoaderModule:
    """Build a dataset and dataloader that can batch variable-length sequences."""

    def __init__(self, eng_data, fre_data, batch_size=32, shuffle=True) -> None:
        """Remember loader settings and create the dataset wrapper."""
        self.eng_data = eng_data
        self.fre_data = fre_data
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        self.dataset = TranslationDataset(eng_data, fre_data)

    def collate_fn(self, batch):
        """Pad a batch so every sequence in the batch has the same length."""
        eng_batch, fre_batch = zip(*batch)

        eng_batch = list(eng_batch)
        fre_batch = list(fre_batch)

        # Pad along the time dimension because sentence lengths vary by sample.
        eng_batch = torch.nn.utils.rnn.pad_sequence(
            eng_batch, padding_value=0
        )

        fre_batch = torch.nn.utils.rnn.pad_sequence(
            fre_batch, padding_value=0
        )

        return eng_batch, fre_batch

    def get_loader(self):
        """Create the PyTorch DataLoader with the custom padding logic attached."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )
