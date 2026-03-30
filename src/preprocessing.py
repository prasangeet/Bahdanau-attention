import pandas as pd 
import re
from collections import Counter

class PreprocessingPipeline:
    """Clean sentence pairs, build vocabularies, and convert text to token ids."""

    def __init__(self, file_path, max_samples=None) -> None:
        """Prepare storage for the dataset, vocab metadata, and sample limit."""
        self.file_path = file_path
        self.max_samples = max_samples
        
        self.eng_vocab = Counter()
        self.fre_vocab = Counter()

        self.eng_word2idx = {}
        self.fre_word2idx = {}

        self.eng_idx2word = {}
        self.fre_idx2word = {}
        
    def clean_text(self, text):
        """Normalize text so the model sees a smaller and cleaner token space."""
        text = text.lower()
        text = re.sub(r"[^a-zA-Z?.!,]+", " ", text)
        return text.strip()

    def load_data(self):
        """Load the CSV file, clean both languages, and add sequence boundary tokens."""
        df = pd.read_csv(self.file_path)

        if self.max_samples is not None:
            # Keep the dataset smaller when we want quicker experiments.
            df = df.head(self.max_samples).copy()

        df.columns = ["English", "French"]

        # Clean both columns with the same lightweight normalization rules.
        df['English'] = df['English'].apply(self.clean_text)
        df['French'] = df['French'].apply(self.clean_text)

        # Give the decoder clear start and end markers for each target sentence.
        df["French"] = "<sos> " + df["French"] + " <eos>"

        self.data = df
        return df
    
    def build_vocab(self):
        """Create word-to-index and index-to-word mappings for both languages."""
        for _, row in self.data.iterrows():
            self.eng_vocab.update(str(row["English"]).split())
            self.fre_vocab.update(str(row["French"]).split())

        # Reserve a few special tokens before adding the regular vocabulary.
        special_tokens = ["<pad>", "<unk>", "<sos>", "<eos>"]

        self.eng_word2idx = {token: i for i, token in enumerate(special_tokens)}
        self.fre_word2idx = {token: i for i, token in enumerate(special_tokens)}

        for word in self.eng_vocab:
            if word not in self.eng_word2idx:
                self.eng_word2idx[word] = len(self.eng_word2idx)

        for word in self.fre_vocab:
            if word not in self.fre_word2idx:
                self.fre_word2idx[word] = len(self.fre_word2idx)

        # Reverse lookups make it easier to inspect predictions later.
        self.eng_idx2word = {i: w for w, i in self.eng_word2idx.items()}
        self.fre_idx2word = {i: w for w, i in self.fre_word2idx.items()}

    def numricalize(self, sentence, vocab):
        """Turn a sentence into token ids, falling back to <unk> for unseen words."""
        return [
            vocab.get(word, vocab["<unk>"])
            for word in sentence.split()
        ]

    def process_data(self):
        """Convert the cleaned dataset into parallel integer sequences."""
        eng_data = []
        fre_data = []

        for _, row in self.data.iterrows():
            # Convert each aligned sentence pair into its numeric form.
            eng_seq = self.numricalize(row["English"], self.eng_word2idx)
            fre_seq = self.numricalize(row["French"], self.fre_word2idx)

            eng_data.append(eng_seq)
            fre_data.append(fre_seq)

        return eng_data, fre_data
