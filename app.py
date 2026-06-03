import argparse
import time

from flask import Flask, jsonify, render_template, request

from agent import (
    FACE_FOR_CLASSIFICATION,
    GIVEUP_REPLY,
    GREETING_REPLY,
    IDENTITY_REPLY,
    NON_CODING_REPLY,
    RUDE_REPLY,
    build_prompt,
    classify_query,
    detect_agent_mode,
)
from config import (
    MAX_HISTORY_TURNS,
    MODELS_DIR,
    available_model_ids,
    default_model_id,
    discover_models,
    get_model,
)
from llm import LLMRegistry, ModelNotAvailable


def create_app() -> Flask:
    app = Flask(__name__)
    registry = LLMRegistry()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/models")
    def list_models():
        models = discover_models()
        return jsonify({
            "models": [
                {"id": mid, "label": cfg["label"], "path": cfg["path"]}
                for mid, cfg in models.items()
            ],
            "default": default_model_id(),
            "models_dir": MODELS_DIR,
        })

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        model_id = data.get("model_id") or default_model_id()
        history = data.get("history") or []

        if not user_message:
            return jsonify({"error": "Empty message", "face_state": "confused"}), 400

        classification = classify_query(user_message, history=history)
        face_state = FACE_FOR_CLASSIFICATION[classification]

        if classification == "rude":
            return jsonify({
                "reply": RUDE_REPLY,
                "mode": "rejected_rude",
                "face_state": face_state,
                "model_id": model_id,
                "remembered": False,
            })

        if classification == "greeting":
            return jsonify({
                "reply": GREETING_REPLY,
                "mode": "greeting",
                "face_state": face_state,
                "model_id": model_id,
                "remembered": False,
            })

        if classification == "identity":
            return jsonify({
                "reply": IDENTITY_REPLY,
                "mode": "identity",
                "face_state": face_state,
                "model_id": model_id,
                "remembered": False,
            })

        if classification == "too_hard":
            return jsonify({
                "reply": GIVEUP_REPLY,
                "mode": "give_up",
                "face_state": face_state,
                "model_id": model_id,
                "remembered": False,
            })

        if classification == "non_coding":
            return jsonify({
                "reply": NON_CODING_REPLY,
                "mode": "rejected_noncoding",
                "face_state": face_state,
                "model_id": model_id,
                "remembered": False,
            })

        if model_id is None or get_model(model_id) is None:
            return jsonify({
                "error": f"No model available. Drop a *.gguf into {MODELS_DIR}.",
                "face_state": "unhappy",
                "model_id": model_id,
            }), 503

        mode = detect_agent_mode(user_message)
        prompt = build_prompt(user_message, mode, history=history, max_history_turns=MAX_HISTORY_TURNS)

        try:
            t0 = time.time()
            reply = registry.generate(model_id, prompt)
            elapsed_ms = int((time.time() - t0) * 1000)
            return jsonify({
                "reply": reply,
                "mode": mode,
                "face_state": "happy",
                "model_id": model_id,
                "elapsed_ms": elapsed_ms,
                "remembered": True,
                "history_turns_used": min(len(history) // 2 if history else 0, MAX_HISTORY_TURNS),
            })
        except ModelNotAvailable as e:
            return jsonify({"error": str(e), "face_state": "unhappy", "model_id": model_id}), 503
        except Exception as e:
            return jsonify({
                "error": f"LLM error: {str(e)}",
                "mode": "error",
                "face_state": "unhappy",
                "model_id": model_id,
            }), 500

    @app.route("/api/benchmark", methods=["POST"])
    def benchmark():
        data = request.get_json(silent=True) or {}
        prompt_text = (data.get("message") or "").strip()
        if not prompt_text:
            return jsonify({"error": "Empty prompt"}), 400
        ids = data.get("model_ids") or available_model_ids()
        mode = detect_agent_mode(prompt_text)
        prompt = build_prompt(prompt_text, mode)

        results = []
        for mid in ids:
            entry = {"model_id": mid}
            try:
                t0 = time.time()
                reply = registry.generate(mid, prompt)
                elapsed = time.time() - t0
                tokens = max(1, len(reply.split()))
                entry.update({
                    "ok": True,
                    "elapsed_ms": int(elapsed * 1000),
                    "approx_tokens": tokens,
                    "approx_tok_per_s": round(tokens / elapsed, 2) if elapsed > 0 else None,
                    "preview": reply.strip().splitlines()[0][:120] if reply.strip() else "",
                })
            except Exception as e:
                entry.update({"ok": False, "error": str(e)})
            results.append(entry)
        return jsonify({"results": results, "mode": mode})

    @app.route("/api/health")
    def health():
        return jsonify({
            "status": "running",
            "backend": "llama.cpp",
            "agent": "CodePi",
            "models_discovered": len(available_model_ids()),
            "default_model": default_model_id(),
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="CodePi — local coding agent")
    parser.add_argument(
        "--mode",
        choices=["web", "cli"],
        default="web",
        help="web = Flask GUI server (default); cli = interactive terminal REPL",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Initial model id (defaults to first discovered in models dir)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "cli":
        from cli import run_cli
        run_cli(args.model)
        return

    app = create_app()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
