import joblib
import re
import pandas as pd

model = joblib.load('news_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text)
    text = re.sub(r'^[a-z\s]+\s*\([a-z\s]+\)\s*', '', text)
    sources = ['reuters', 'associated press', 'ap news', 'cnn', 'fox news', 'bbc', 'politico', 'nbc news', 'cbs news', 'the atlantic', 'washington post', 'nytimes', 'new york times']
    for source in sources:
        text = text.replace(source, '')
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

df = pd.read_csv('news.csv')
print(f"{'TITLE':<30} | {'PRED':<6} | {'EXPECTED':<6}")
print("-" * 50)
for i, row in df.iterrows():
    content = clean_text(str(row['title']) + " " + str(row['text']))
    vec = vectorizer.transform([content])
    pred = model.predict(vec)[0]
    print(f"{str(row['title'])[:30]:<30} | {pred:<6} | {row['label']:<6}")
