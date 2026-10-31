from collections import Counter

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import nltk
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


# Dataset has duplicate categories mapping each
CATEGORY_MERGES = {
    "THE WORLDPOST": "WORLD NEWS",
    "WORLDPOST": "WORLD NEWS",
    "ARTS": "ARTS & CULTURE",
    "CULTURE & ARTS": "ARTS & CULTURE",
    "PARENTS": "PARENTING",
    "STYLE": "STYLE & BEAUTY",
    "GREEN": "ENVIRONMENT",
    "TASTE": "FOOD & DRINK",
    "HEALTHY LIVING": "WELLNESS",
    "COLLEGE": "EDUCATION",
}


class CustomTextDataset(Dataset):
    def __init__(self, csv_file, verbose=False, use_description=False, max_len=None, min_freq=1, 
                 merge_categories=False):
        
        #Load data 
        if str(csv_file).endswith(".json"):

            # read in chunks cuz my computer is weak :( so memory issues 
            data = pd.concat(pd.read_json(csv_file, lines=True, chunksize=20000), ignore_index=True)
        else:
            data = pd.read_csv(csv_file)

        self.link = data["link"]
        self.headline = data["headline"].fillna("")
        self.category = data["category"]
        if merge_categories:
            self.category = self.category.replace(CATEGORY_MERGES)
        self.short_desc = data["short_description"]
        self.authors = data["authors"].fillna("")
        self.date = data["date"].astype(str)

        # Convert variables into int
        encoder_category = LabelEncoder()
        encoder_authors = LabelEncoder()
        encoder_date = LabelEncoder()

        self.category_int = encoder_category.fit_transform(self.category)
        self.authors_int = encoder_authors.fit_transform(self.authors)
        self.date_int = encoder_date.fit_transform(self.date)

        self.class_names = list(encoder_category.classes_)

        # Summery of category list
        if verbose:
            print(f"{len(encoder_category.classes_)} categories, "
                  f"{len(encoder_authors.classes_)} authors, "
                  f"{len(encoder_date.classes_)} dates")

        # Preprocessing, text tokenisation, lemmatisation etc
        stop_words = set(stopwords.words('english'))

        tokeniser = RegexpTokenizer(r'\w+')
        lemmatiser = WordNetLemmatizer()
        lemma_cache = {}  # for speed

        if use_description:
            texts = self.headline + " " + self.short_desc.fillna("")
        else:
            texts = self.headline

        self.headline_tokenised = []
        for headline in texts:
            headline = headline.lower()

            words = tokeniser.tokenize(headline)

            tokenised = []
            for word in words:
                if word not in stop_words:
                    tokenised.append(word)

            lemmatised_tokens = []
            for word in tokenised:
                if word not in lemma_cache:
                    lemma_cache[word] = lemmatiser.lemmatize(word)
                lemmatised_tokens.append(lemma_cache[word])

            # Truncate long text
            if max_len is not None:
                lemmatised_tokens = lemmatised_tokens[:max_len]

            self.headline_tokenised.append(lemmatised_tokens)

        # Add EOS and PAD tokens
        self.max_sentence_len = 0
        for headline in self.headline_tokenised:
            self.max_sentence_len = max(len(headline), self.max_sentence_len)

        for i in range(len(self.headline_tokenised)):
            self.headline_tokenised[i].append("<EOS>")
            pad = self.max_sentence_len + 1 - len(self.headline_tokenised[i])

            self.headline_tokenised[i].extend(["<PAD>"] * pad)

        ## Build vocabulary

        # PAD at 0 so ignored, UNK at 1 only for less frequent words since 200k article dataset
        counts = Counter(token for headline in self.headline_tokenised for token in headline)
        self.unique_tokens = ["<PAD>", "<UNK>"]
        for token, count in counts.items():
            if count >= min_freq and token not in ("<PAD>", "<UNK>"):
                self.unique_tokens.append(token)
        self.pad_idx = 0
        self.unk_idx = 1

        # Create dictionary
        word_dict = {token: idx for idx, token in enumerate(self.unique_tokens)}


        self.token_ids = np.array(
            [[word_dict.get(token, self.unk_idx) for token in headline] for headline in self.headline_tokenised],
            dtype=np.int64,
        )

        # Convert to torch tensors
        self.token_ids = torch.as_tensor(self.token_ids)
        del self.headline_tokenised
        self.category_int = torch.as_tensor(self.category_int)
        self.authors_int = torch.as_tensor(self.authors_int)
        self.date_int = torch.as_tensor(self.date_int)

    def __len__(self):
        return len(self.category_int)

    def __getitem__(self, idx):
        headline = self.token_ids[idx]
        target = self.category_int[idx]
        sample = [headline, target]
        return sample
