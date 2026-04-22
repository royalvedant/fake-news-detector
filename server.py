from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from newspaper import Article, Config
import numpy as np
import os
import re
import nltk

# Ensure NLTK data is present for newspaper3k
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

app = Flask(__name__)
CORS(app)

# Load existing ML assets
MODEL_PATH = 'news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None

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

def calculate_confidence(decision_score):
    """
    Convert the distance from the hyperplane (decision_function) 
    into a human-readable certainty percentage.
    """
    # Using a sigmoid-like mapping to scale [0, inf] to [50, 100]
    # abs(score) of 0 = 50% (uncertain), abs(score) > 3 = ~95%+
    prob = 1 / (1 + np.exp(-abs(decision_score)))
    return int(prob * 100)

@app.route('/detect', methods=['POST'])
def detect():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not trained or assets missing'}), 500

    data = request.get_json()
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    try:
        # 1. Scrape content using newspaper3k with robust headers
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        config.request_timeout = 10
        
        article = Article(url, config=config)
        article.download()
        article.parse()

        headline = article.title
        body_text = article.text
        
        # 2. Scrape Quality Check & Fallback
        # If the body text is blocked or suspicious, we prioritize the headline
        warning = None
        if not body_text or len(body_text.strip()) < 250:
            full_content = clean_text(headline)
            warning = "Analysis based on headline only (Website blocked full text scraping)."
        else:
            full_content = clean_text(f"{headline} {body_text}")

        # 3. Vectorize and Predict
        tfidf_input = vectorizer.transform([full_content])
        prediction = model.predict(tfidf_input)[0]

        # 3. Calculate Confidence
        # Decision function gives the distance to the hyperplane
        raw_score = model.decision_function(tfidf_input)[0]
        # Map raw score to a 0-100 percentage (approximate)
        confidence = min(int(abs(raw_score) * 10), 100) if abs(raw_score) < 10 else 100
        
        # Length bias mitigation: If the article is too short, the AI is naturally less certain
        if not warning and len(headline + body_text) < 200:
            confidence = int(confidence * 0.7) # 30% penalty for short text
            warning = "Article too short for high-accuracy analysis."
        
        return jsonify({
            "label": prediction,
            "confidence": confidence,
            "warning": warning,
            "headline": headline,
            "text_sample": body_text[:200] + "..." if len(body_text) > 200 else body_text
        })

    except Exception as e:
        return jsonify({'error': f'Failed to process URL: {str(e)}'}), 500

if __name__ == '__main__':
    # Running on Port 5000 as per React frontend expectation
    app.run(port=5000, debug=True)
