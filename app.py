from flask import Flask, request, jsonify, render_template
from chatbot import get_response
from datetime import datetime

app = Flask(__name__)

# ── Simple in-memory analytics (resets on restart) ──────────────
chat_logs = []

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/chat")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/chat-api", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    result = get_response(user_message)

    # Log the conversation
    chat_logs.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_message,
        "bot": result["answer"],
        "confidence": result["confidence"]
    })

    return jsonify({
        "reply": result["answer"],
        "confidence": result["confidence"],
        "matched_question": result["matched_question"]
    })


@app.route("/stats")
def stats():
    return jsonify({
        "total_queries": len(chat_logs),
        "recent": chat_logs[-10:] if chat_logs else []
    })


if __name__ == "__main__":
    print("🏦 EaseBanking.AI server starting...")
    print("🌐 Open http://127.0.0.1:5000 in your browser")
    print("📊 Admin: http://127.0.0.1:5000/admin")
    app.run(debug=True)