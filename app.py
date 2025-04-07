from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle, requests, re, nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f: model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'\W', ' ', text).lower().split()
    return ' '.join([w for w in text if w not in stop_words])

def get_real_news_sources():
    try:
        response = requests.get("https://newsapi.org/v2/top-headlines?country=us&apiKey=71ec87d1341249c7946deaae98d4b0ce")
        articles = response.json().get("articles", [])
        return [{"source": a["source"]["name"], "url": a["url"]} for a in articles[:5]]
    except:
        return [{"error": "Could not fetch sources"}]

@app.route('/')
def home():
    return "✅ Fake News Detection API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'news_text' not in data:
        return jsonify({"error": "Missing news_text"}), 400

    text = clean_text(data['news_text'])
    vect = vectorizer.transform([text])
    prediction = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0][1]  # probability of Real News
    confidence = round(prob * 100, 2)

    result = "Real News" if prediction == 1 else "Fake News"
    sources = get_real_news_sources() if prediction == 1 else []

    return jsonify({
        "prediction": result,
        "confidence": confidence,
        "sources": sources
    })

if __name__ == '__main__':
    app.run(debug=True)