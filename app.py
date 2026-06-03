from flask import Flask, jsonify, render_template, request

from agent import build_prompt, detect_agent_mode, is_coding_question
from llm import LocalLLM

app = Flask(__name__)
llm = LocalLLM()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    if not is_coding_question(user_message):
        return jsonify({
            "reply": "I am CodePi, a coding-only AI assistant. Please ask a programming-related question.",
            "mode": "rejected",
        })

    mode = detect_agent_mode(user_message)
    prompt = build_prompt(user_message, mode)

    try:
        reply = llm.generate(prompt)
        return jsonify({"reply": reply, "mode": mode})
    except Exception as e:
        return jsonify({"error": f"LLM error: {str(e)}", "mode": "error"}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "running", "backend": "llama.cpp", "agent": "CodePi"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
