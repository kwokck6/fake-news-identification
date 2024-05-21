import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# read csv files and clean datasets if necessary
true_df = pd.read_csv('data/True.csv')
fake_df1 = pd.read_csv('data/Fake.csv')
fake_df2 = pd.read_csv('data/fake_bs.csv', encoding_errors='backslashreplace')
print(f'Original # samples: {len(true_df) + len(fake_df1) + len(fake_df2)}')
fake_df2 = fake_df2[fake_df2['language'] == 'english']
fake_df2.loc[fake_df2['text'].isna(), 'language'] = 'french'
fake_df2 = fake_df2[fake_df2['language'] == 'english']


# set labels
true_df['label'] = 1
fake_df1['label'] = 0
fake_df2['label'] = 0

# join the datasets together
df = pd.concat([true_df, fake_df1, fake_df2])

text_preprocess = lambda s: ' '.join(s.translate(s.maketrans('-\'\"’“”\t', '       ', '!#$%^&*()[]\{\}\"<>?,.;:\\~+=0123456789')).lower().split())  # remove punctuation marks and numbers
print(f'# Empty titles: {len(df[df["title"].isna()])}')
print(f'# Empty articles: {len(df[df["text"].isna()])}')
df['text'] = df['text'].fillna('').apply(text_preprocess)  # avoid being removed by dropna
df['title'] = df['title'].fillna('').apply(text_preprocess)  # avoid being removed by dropna
df.dropna(axis=1, inplace=True)  # remove uncommon rows loaded with NaNs
print(f'Total # samples: {len(df)}')


# count length of articles before truncation
text_before = pd.DataFrame({'tokens': df['text'].astype(str).apply(lambda x: x.split())})
text_before['length'] = text_before['tokens'].apply(len)
fig = sns.histplot(text_before['length'])
plt.title('Length of articles before truncation')
plt.savefig('len-of-text-before.png')
plt.clf()
print(f'TEXT Before truncation - Maximum # words: {text_before["length"].max()} | Minimum # words: {text_before["length"].min()}')
print(f"Shortest article: {' '.join(text_before.iloc[text_before['length'].argmin()]['tokens'])}\nLongest article: {' '.join(text_before.iloc[text_before['length'].argmax()]['tokens'][:10]) + ' ... ' + ' '.join(text_before.iloc[text_before['length'].argmax()]['tokens'][-11: -1])}")
print(text_before['length'].describe())

# count length of articles after truncation
threshold = 256
text_after = pd.DataFrame({'tokens': df['text'].astype(str).apply(lambda x: x.split(maxsplit=threshold))})
text_after['length'] = text_after['tokens'].apply(len)
fig = sns.histplot(text_after['length'])
plt.title('Length of articles after truncation')
plt.savefig('len-of-text-after.png')
plt.clf()
print(f'TEXT After truncation - Maximum # words: {text_after["length"].max()} | Minimum # words: {text_after["length"].min()}')
print(f"Shortest article: {' '.join(text_after.iloc[text_after['length'].argmin()]['tokens'])}\nFirst longest article: {' '.join(text_after.iloc[text_after['length'].argmax()]['tokens'][:10]) + ' ... ' + ' '.join(text_after.iloc[text_after['length'].argmax()]['tokens'][-11: -1])}")
print(text_after['length'].describe())

# count length of title
title = pd.DataFrame({'tokens': df['title'].astype(str).apply(lambda x: x.split())})
title['length'] = title['tokens'].apply(len)
fig = sns.histplot(title['length'])
plt.title('Length of titles')
plt.savefig('len-of-title-before.png')
plt.clf()
print(f'TITLE - Maximum # words: {title["length"].max()} | Minimum # words: {title["length"].min()}')
print(f"Shortest title: {' '.join(title.iloc[title['length'].argmin()]['tokens'])}\nFirst longest title: {' '.join(title.iloc[title['length'].argmax()]['tokens'][:10]) + ' ... ' + ' '.join(title.iloc[title['length'].argmax()]['tokens'][-11: -1])}")
print(title['length'].describe())

# count samples grouped by classes
fig = sns.countplot(data=df, x='label')
plt.title('Number of news articles')
plt.savefig('class_count.png')
plt.clf()

# save csv
df.to_csv('data/processed.csv', index=False)
