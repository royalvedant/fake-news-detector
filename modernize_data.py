import requests
import pandas as pd
import time
from newspaper import Article

API_KEY = "b2b75db3c6fc44f28565992a46f7a413"

def fetch_modern_news():
    print("Fetching modern REAL news from NewsAPI...")
    # Get recent news about India using 'everything' endpoint
    url = f"https://newsapi.org/v2/everything?q=India&language=en&pageSize=100&sortBy=publishedAt&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    modern_data = []
    for art in articles:
        if art['url'] and art['description']:
            try:
                # We use the title + description for speed, or scrape full text if needed
                # For modernization, title + description is often enough to learn modern tokens
                modern_data.append({
                    "text": f"{art['title']} {art['description']}",
                    "label": "REAL"
                })
            except Exception:
                continue
                
    return pd.DataFrame(modern_data)

def fetch_modern_satire():
    print("Adding modern 'Style' tokens (Satire/Opinion)...")
    # We'll use a few known modern satirical headlines to teach the AI what modern FAKE looks like
    # (Since we can't easily scrape them all now, we'll use a high-quality manual sample)
    satire_samples = [
        {"text": "Modern AI Overlord promises to keep humans as pets in spacious climate-controlled biodomes.", "label": "FAKE"},
        {"text": "Local man discovers secret to eternal life is just eating one grape per hour for 80 years.", "label": "FAKE"},
        {"text": "2026 Elections: Hologram of George Washington enters the race with 'Ghost of Liberty' platform.", "label": "FAKE"},
        {"text": "Tech Giant 'X' announces plan to replace all physical currency with 'Thoughts and Prayers' digital tokens.", "label": "FAKE"},
        {"text": "NASA confirms the moon is actually a giant disco ball left by an ancient space-faring civilization.", "label": "FAKE"}
    ]
    return pd.DataFrame(satire_samples)

if __name__ == "__main__":
    real_df = fetch_modern_news()
    fake_df = fetch_modern_satire()
    
    combined = pd.concat([real_df, fake_df], ignore_index=True)
    combined.to_csv('modern_data.csv', index=False)
    print(f"Modernized Data Saved: {len(combined)} rows")
