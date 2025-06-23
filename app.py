from flask import Flask, render_template, request, redirect, url_for, session
import os, json
import google.generativeai as genai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = "bike_secret"

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT_FILE = os.path.join(BASE_DIR, "chatbot_data.txt")
USERS_FILE = os.path.join(BASE_DIR, "users.json")

# Configure Gemini
genai.configure(api_key="AIzaSyCHwKH26j5Fivra85njdnnPxK5RtQDJyV0")

# Load users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Load chatbot Q&A pairs
def load_chatbot_data():
    pairs = []
    try:
        with open(CHATBOT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if ":" in line:
                    q, a = line.strip().split(":", 1)
                    pairs.append((q.strip(), a.strip()))
    except FileNotFoundError:
        return []
    return pairs

# Generate chatbot response using Gemini embeddings
def get_response(user_input):
    data = load_chatbot_data()
    if not data:
        return "Chatbot data missing."

    questions = [q for q, _ in data]
    responses = [a for _, a in data]

    try:
        user_embed = genai.embed_content(
            model="models/embedding-001",
            content=user_input,
            task_type="retrieval_query"
        )["embedding"]

        q_embeddings = [
            genai.embed_content(
                model="models/embedding-001",
                content=q,
                task_type="retrieval_document"
            )["embedding"] for q in questions
        ]

        sims = cosine_similarity([user_embed], q_embeddings)[0]
        best_idx = int(np.argmax(sims))

        if sims[best_idx] < 0.7:
            return "Sorry, I couldn't find a relevant answer."

        return responses[best_idx]

    except Exception as e:
        return f"Error: {e}"

# Routes
@app.route("/", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    user_input = ""
    bot_response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        bot_response = get_response(user_input)
    return render_template("index.html", user_input=user_input, bot_response=bot_response)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]
        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("index"))
        return "Invalid username or password"
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        users = load_users()
        username = request.form["username"]
        password = request.form["password"]
        if username in users:
            return "Username already exists"
        users[username] = password
        save_users(users)
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
