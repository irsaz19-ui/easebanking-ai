# 🏦 EaseBanking.AI — FAQ Chatbot

A Python-based AI banking FAQ chatbot built with Flask + TF-IDF NLP.

---

## 📁 Project Structure

```
easebanking-ai/
├── app.py              ← Flask web server
├── chatbot.py          ← NLP chatbot engine (TF-IDF)
├── requirements.txt    ← Python dependencies
├── data/
│   └── faqs.json       ← 30 banking FAQ entries
└── templates/
    └── index.html      ← Chat UI (served by Flask)
```

---

## ⚙️ Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the chatbot in terminal (optional test)
```bash
python chatbot.py
```

### 3. Start the web server
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## 🤖 How It Works

1. User types a banking question
2. The text is preprocessed (lowercased, stopwords removed, stemmed)
3. TF-IDF vectorizer converts it to a numeric vector
4. Cosine similarity finds the closest FAQ match
5. If confidence > 20%, the answer is returned
6. Otherwise, a fallback message with support contacts is shown

---

## 🧪 Test Questions

- "How do I check my balance?"
- "What is the UPI limit?"
- "How to block my debit card?"
- "What documents are needed for KYC?"
- "What are the FD interest rates?"

---

## 📈 Possible Enhancements (Week 4)

- [ ] Add OpenAI GPT for smarter answers
- [ ] Admin dashboard for analytics
- [ ] Multi-language support (Hindi)
- [ ] Voice input/output
- [ ] User authentication

---

## 👨‍💻 Built With

- Python 3.10+
- Flask
- scikit-learn (TF-IDF + Cosine Similarity)
- NLTK
- HTML / CSS / JavaScript
