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
    BERT_MODEL_NAME,
    CONTEXT_SAFETY,
    ELO_STATE_PATH,
    MAX_HISTORY_TURNS,
    MAX_TOKENS,
    MODEL_CONTEXT_SIZE,
    MODELS_DIR,
    RAG_EMBEDDINGS_PATH,
    RAG_META_PATH,
    available_model_ids,
    default_model_id,
    discover_models,
    get_model,
)
from llm import LLMRegistry, ModelNotAvailable
from routing import EloStore, Router, update_elo


def create_app() -> Flask:
    app = Flask(__name__)
    registry = LLMRegistry()
    elo_store = EloStore(ELO_STATE_PATH)
    router = Router(RAG_EMBEDDINGS_PATH, RAG_META_PATH, BERT_MODEL_NAME)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/models")
    def list_models():
        models = discover_models()
        return jsonify({
            "models": [
                {"id": mid, "label": cfg["label"], "path": cfg["path"],
                 "elo": elo_store.get(mid)}
                for mid, cfg in models.items()
            ],
            "default": default_model_id(),
            "models_dir": MODELS_DIR,
            "auto_supported": True,
            "router": {
                "using_bert": router.using_bert,
                "load_error": router.load_error,
            },
        })

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json(silent=True) or {}
        user_message = (data.get("message") or "").strip()
        requested_model_id = data.get("model_id")
        history = data.get("history") or []

        if not user_message:
            return jsonify({"error": "Empty message", "face_state": "confused"}), 400

        classification = classify_query(user_message, history=history)
        face_state = FACE_FOR_CLASSIFICATION[classification]

        # Echo the requested model_id back on rejection replies (these don't
        # invoke the LLM, so no routing happens for them).
        rejection_model_id = requested_model_id or default_model_id()

        if classification == "rude":
            return jsonify({
                "reply": RUDE_REPLY,
                "mode": "rejected_rude",
                "face_state": face_state,
                "model_id": rejection_model_id,
                "remembered": False,
            })

        if classification == "greeting":
            return jsonify({
                "reply": GREETING_REPLY,
                "mode": "greeting",
                "face_state": face_state,
                "model_id": rejection_model_id,
                "remembered": False,
            })

        if classification == "identity":
            return jsonify({
                "reply": IDENTITY_REPLY,
                "mode": "identity",
                "face_state": face_state,
                "model_id": rejection_model_id,
                "remembered": False,
            })

        if classification == "too_hard":
            return jsonify({
                "reply": GIVEUP_REPLY,
                "mode": "give_up",
                "face_state": face_state,
                "model_id": rejection_model_id,
                "remembered": False,
            })

        if classification == "non_coding":
            return jsonify({
                "reply": NON_CODING_REPLY,
                "mode": "rejected_noncoding",
                "face_state": face_state,
                "model_id": rejection_model_id,
                "remembered": False,
            })

        # Routing: model_id "auto" or missing -> ask the router. An explicit
        # model_id from the dropdown short-circuits the router.
        ids = available_model_ids()
        routing_info = None
        if requested_model_id and requested_model_id != "auto":
            model_id = requested_model_id
        else:
            routing_info = router.route(user_message, ids, elo_store)
            model_id = routing_info.get("chosen_model_id") or default_model_id()

        if model_id is None or get_model(model_id) is None:
            return jsonify({
                "error": f"No model available. Drop a *.gguf into {MODELS_DIR}.",
                "face_state": "unhappy",
                "model_id": model_id,
                "routing": routing_info,
            }), 503

        mode = detect_agent_mode(user_message)
        prompt_budget = max(256, MODEL_CONTEXT_SIZE - MAX_TOKENS - CONTEXT_SAFETY)
        prompt = build_prompt(
            user_message, mode,
            history=history,
            max_history_turns=MAX_HISTORY_TURNS,
            max_prompt_tokens=prompt_budget,
        )

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
                "model_elo": elo_store.get(model_id),
                "routing": routing_info,
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

    @app.route("/api/route", methods=["POST"])
    def route_only():
        """Dry-run the router: returns chosen model + predicted difficulty."""
        data = request.get_json(silent=True) or {}
        msg = (data.get("message") or "").strip()
        if not msg:
            return jsonify({"error": "empty message"}), 400
        info = router.route(msg, available_model_ids(), elo_store)
        return jsonify(info)

    @app.route("/api/vote", methods=["POST"])
    def vote():
        """Apply an ELO round to a model based on a thumbs up/down vote."""
        data = request.get_json(silent=True) or {}
        mid = data.get("model_id")
        vote_dir = data.get("vote")
        target_elo = data.get("target_elo")
        if not mid or vote_dir not in ("up", "down"):
            return jsonify({"error": "need model_id and vote in {up,down}"}), 400
        if target_elo is None:
            target_elo = elo_store.get(mid)
        new_elo = update_elo(elo_store, mid, int(target_elo), vote_dir)
        return jsonify({
            "model_id": mid,
            "vote": vote_dir,
            "target_elo": int(target_elo),
            "new_elo": new_elo,
        })

    @app.route("/api/elo")
    def elo_dump():
        models = discover_models()
        return jsonify({
            "elo": {mid: elo_store.get(mid) for mid in models},
            "raw_overrides": elo_store.all(),
        })

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
