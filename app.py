import streamlit as st
import requests
import joblib
import pandas as pd
from newspaper import Article
import re
import nltk

# --- NLTK SETUP ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# --- CONFIGURATION ---
API_KEY = "b2b75db3c6fc44f28565992a46f7a413"
MODEL_PATH = 'news_model.pkl'
VECTOR_PATH = 'tfidf_vectorizer.pkl'

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTOR_PATH)
    return model, vectorizer

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    # 1. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 2. Remove standard news agency signatures
    text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text)
    text = re.sub(r'^[a-z\s]+\s*\([a-z\s]+\)\s*', '', text)
    # 3. Remove source names globally to prevent bias shortcuts
    sources = ['reuters', 'associated press', 'ap news', 'cnn', 'fox news', 'bbc', 'politico', 'nbc news', 'cbs news', 'the atlantic', 'washington post', 'nytimes', 'new york times']
    for source in sources:
        text = text.replace(source, '')
    # 4. Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# --- HELPER FUNCTIONS ---
def fetch_news(query="top-headlines"):
    url = f'https://newsapi.org/v2/top-headlines?country=us&apiKey={API_KEY}'
    if query != "top-headlines":
        url = f'https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}'
    try:
        response = requests.get(url)
        return response.json().get('articles', [])
    except Exception:
        return []

def predict_news(text, model, vectorizer):
    if not text or text.strip() == "":
        return None
    cleaned_text = clean_text(text)
    vec_input = vectorizer.transform([cleaned_text])
    prediction = model.predict(vec_input)[0]
    return prediction

# --- UI SETUP ---
st.set_page_config(page_title="AI News Guard Pro", layout="wide")
st.title("📡 AI News Guard: Real-Time Fake News Detector")
st.markdown("---")

model, vectorizer = load_assets()

# Tabbed Interface
tab1, tab2 = st.tabs(["📊 Live Headlines", "🔗 Verify by URL"])

# --- TAB 1: LIVE HEADLINES ---
with tab1:
    st.header("Sync Real-Time News")
    st.sidebar.header("Live Feed Settings")
    search_query = st.sidebar.text_input("Search Specific Topic:", placeholder="e.g. Election, Tech")
    sync_btn = st.sidebar.button("Sync Live News")

    if sync_btn or search_query:
        query = search_query if search_query else "top-headlines"
        with st.spinner('Fetching latest news...'):
            articles = fetch_news(query)
        
        if not articles:
            st.warning("No news found for this topic.")
        
        for art in articles:
            content = f"{art['title']} {art['description'] or ''}"
            prediction = predict_news(content, model, vectorizer)
            if prediction is None: continue

            with st.container():
                col1, col2 = st.columns([1, 4])
                with col1:
                    if prediction == "REAL":
                        st.success("✅ REAL")
                    else:
                        st.error("🚩 FAKE")
                with col2:
                    st.subheader(art['title'])
                    st.caption(f"Source: {art['source']['name']} | Published: {art['publishedAt']}")
                    st.write(art['description'])
                    st.markdown(f"[Read Article]({art['url']})")
                st.divider()
    else:
        st.info("Click 'Sync Live News' or enter a search term in the sidebar to view the live feed.")

# --- TAB 2: VERIFY BY URL ---
with tab2:
    st.header("Analyze Any News Link")
    input_url = st.text_input("Enter Website URL:", placeholder="https://news-site.com/article-link")
    detect_btn = st.button("Analyze Link")

    if detect_btn and input_url:
        try:
            with st.spinner('Scraping and analyzing content...'):
                article = Article(input_url)
                article.download()
                article.parse()
                
                headline = article.title
                body_text = article.text

                prediction = predict_news(f"{headline} {body_text}", model, vectorizer)

                st.subheader(f"Extracted Headline: {headline}")
                if prediction == "REAL":
                    st.success("✅ This news appears to be REAL.")
                else:
                    st.error("🚩 WARNING: This news is likely FAKE.")
                
                with st.expander("Show extracted text"):
                    st.write(body_text)
        except Exception as e:
            st.error(f"Error processing URL: {e}")
