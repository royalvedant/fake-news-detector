import joblib
import re
from newspaper import Article, Config

MODEL_PATH = 'news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

url = "https://timesofindia.indiatimes.com/defence/news/india-does-not-forget-army-sends-strong-message-on-pahalgam-terror-attack-anniversary-recalls-op-sindoor/articleshow/130407984.cms"

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

print(f"Scraping URL with Custom User-Agent: {url}")
try:
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    
    article = Article(url, config=config)
    article.download()
    article.parse()
    
    headline = article.title
    body_text = article.text
    content = clean_text(f"{headline} {body_text}")
    
    print("-" * 30)
    print(f"Headline: {headline}")
    print(f"Text Length: {len(body_text)} chars")
    print(f"Extracted Text (first 300 chars):\n{body_text[:300]}...")
    print("-" * 30)

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    vec = vectorizer.transform([content])
    pred = model.predict(vec)[0]
    raw_score = model.decision_function(vec)[0]
    
    print(f"PREDICTION: {pred}")
    print(f"Decision Function Score: {raw_score}")

except Exception as e:
    print(f"Error: {e}")
