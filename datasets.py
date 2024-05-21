import pandas as pd
from torch.utils.data import Dataset


class FakeNews(Dataset):
    def __init__(self, split=None, text_transform=None, target_transform=None):
        self.df = self.split_df(split)
        self.text_transform = text_transform
        self.target_transform = target_transform
        self.num_classes = 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.df.iloc[index]['text'], self.df.iloc[index]['label']

    def split_df(self, split):
        df = self.read_df()
        seed = 230412
        if split is None:
            return df

        train, val, test = 0.7, 0.15, 0.15
        train_df = df.sample(frac=train, random_state=seed)
        if split == 'train':
            return train_df
        
        val_df = df.drop(train_df.index).sample(frac=val/(1-train), random_state=seed)
        if split == 'val':
            return val_df
        
        test_df = df.drop(train_df.index).drop(val_df.index)
        if split == 'test':
            return test_df

    def read_df(self):
        df = pd.read_csv('data/processed.csv')
        # df['text'].fillna('', inplace=True)
        # df['title'].fillna('', inplace=True)
        df = df[~df['text'].isna()]
        return df


if __name__ == '__main__':
    dataset = FakeNews(split='val')
    print(dataset.df.head())
