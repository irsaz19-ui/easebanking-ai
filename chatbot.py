import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re
import os

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ─────────────────────────────────────────
# Load FAQ Data
# ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAQ_PATH = os.path.join(BASE_DIR, "data", "faqs.json")

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_data = json.load(f)

questions = [item["question"] for item in faq_data]
answers   = [item["answer"]   for item in faq_data]

# ─────────────────────────────────────────
# NLP Preprocessing
# ─────────────────────────────────────────
stemmer    = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

processed_questions = [preprocess(q) for q in questions]

# ─────────────────────────────────────────
# TF-IDF Vectorizer
# ─────────────────────────────────────────
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(processed_questions)

# ─────────────────────────────────────────
# Greeting & Small Talk
# ─────────────────────────────────────────
GREETINGS = {
    "hi": "Hello! 👋 Welcome to EaseBanking.AI. How can I assist you today?",
    "hello": "Hi there! 😊 I'm your EaseBanking virtual assistant. What can I help you with?",
    "hey": "Hey! 👋 I'm here to help with all your banking queries!",
    "good morning": "Good morning! ☀️ How can I assist you with your banking needs today?",
    "good afternoon": "Good afternoon! 🌤️ What banking help do you need today?",
    "good evening": "Good evening! 🌙 How can I assist you?",
    "how are you": "I'm doing great, thank you for asking! 😊 I'm always ready to help with your banking queries.",
    "thanks": "You're welcome! 😊 Feel free to ask if you have more questions.",
    "thank you": "Happy to help! 🙏 Is there anything else you'd like to know?",
    "bye": "Goodbye! 👋 Have a great day and stay safe. Visit us again!",
    "goodbye": "Take care! 👋 EaseBanking is always here for you.",
    "help": "I can help you with account queries, transfers, loans, cards, KYC, interest rates, and more. Just type your question!",
}

def check_greeting(user_input: str):
    cleaned = user_input.lower().strip().rstrip("!?.")
    for key, response in GREETINGS.items():
        if key in cleaned:
            return response
    return None

# ─────────────────────────────────────────
# Main Chat Function
# ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.20

def get_response(user_input: str) -> dict:
    """
    Returns a dict with:
      - answer  : str
      - confidence : float (0–1)
      - matched_question : str | None
    """
    if not user_input or not user_input.strip():
        return {
            "answer": "Please type a question so I can help you! 😊",
            "confidence": 0.0,
            "matched_question": None
        }

    # Check greetings / small talk first
    greeting_response = check_greeting(user_input)
    if greeting_response:
        return {
            "answer": greeting_response,
            "confidence": 1.0,
            "matched_question": None
        }

    # Vectorise user query
    processed_input = preprocess(user_input)
    user_vec = vectorizer.transform([processed_input])

    # Cosine similarity against all FAQ questions
    similarities = cosine_similarity(user_vec, tfidf_matrix).flatten()
    best_idx     = int(np.argmax(similarities))
    best_score   = float(similarities[best_idx])

    if best_score >= CONFIDENCE_THRESHOLD:
        return {
            "answer": answers[best_idx],
            "confidence": round(best_score, 4),
            "matched_question": questions[best_idx]
        }
    else:
        return {
            "answer": (
                "I'm sorry, I couldn't find a specific answer to your query. 🤔\n\n"
                "You can:\n"
                "📞 Call us: 1800-XXX-0000 (24x7 Toll-Free)\n"
                "📧 Email: support@easebanking.ai\n"
                "🌐 Visit: www.easebanking.ai/support\n\n"
                "Is there anything else I can help you with?"
            ),
            "confidence": round(best_score, 4),
            "matched_question": None
        }


# ─────────────────────────────────────────
# CLI Test (run directly)
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  🏦  EaseBanking.AI — FAQ Chatbot  ")
    print("  Type 'quit' to exit")
    print("=" * 55)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bot: Goodbye! Have a great day! 👋")
            break
        result = get_response(user_input)
        print(f"\nBot: {result['answer']}")
        if result["matched_question"]:
            print(f"     [Matched: '{result['matched_question']}' | Score: {result['confidence']}]")
