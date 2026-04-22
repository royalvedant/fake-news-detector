import joblib
import re

MODEL_PATH = 'news_model.pkl'
VECTORIZER_PATH = 'tfidf_vectorizer.pkl'

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def clean_text(text):
    text = re.sub(r'^.*?\(Reuters\)\s*-\s*', '', str(text))
    text = re.sub(r'^[A-Z\s]+\s*\([A-Z\s]+\)\s*', '', text)
    text = re.sub(r'\s*-\s*(CBS News|Politico|NBC News|AP News|BBC|Fox News|CNN|Ars Technica|Variety|NPR|The Atlantic|The Washington Post|9to5Mac|Android Authority).*$', '', text, flags=re.IGNORECASE)
    return text.strip()

samples = [
    {
        "title": "Fed chair nominee Warsh set to commit to be ‘strictly independent’ on rates",
        "text": "Kevin Warsh will underscore his commitment to keeping inflation in check, saying price stability is a mandate for the Fed “without excuse or equivocation, argument or anguish.” - Politico",
        "expected": "REAL"
    },
    {
        "title": "NASA confirms the moon is actually a giant disco ball",
        "text": "NASA confirms the moon is actually a giant disco ball left by an ancient space-faring civilization.",
        "expected": "FAKE"
    }
]

for s in samples:
    content = clean_text(f"{s['title']} {s['text']}")
    vec = vectorizer.transform([content])
    pred = model.predict(vec)[0]
    print(f"Title: {s['title']}")
    print(f"Prediction: {pred} (Expected: {s['expected']})")
    print("-" * 20)
