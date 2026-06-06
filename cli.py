import time

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
    ELO_STATE_PATH,
    MAX_HISTORY_TURNS,
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

C_RESET = "\033[0m"
C_DIM = "\033[2m"
C_BOLD = "\033[1m"
C_CYAN = "\033[36m"
C_GREEN = "\033[32m"
C_YELLOW = "\033[33m"
C_RED = "\033[31m"
C_MAGENTA = "\033[35m"


BANNER = r"""
   ____           _      ____  _
  / ___|___   __| | ___|  _ \(_)
 | |   / _ \ / _` |/ _ \ |_) | |
 | |__| (_) | (_| |  __/  __/| |
  \____\___/ \__,_|\___|_|   |_|
  Local coding agent  ::  CLI mode
"""

HELP = """
Commands:
  :help            show this help
  :models          list models discovered in the models directory
  :use <model_id>  switch active model (turns auto routing OFF)
  :auto on|off     turn router on/off (BERT + LeetCode RAG)
  :route <prompt>  dry-run the router; show chosen model and difficulty
  :elo             show current ELO for every discovered model
  :vote up|down    vote on the last reply (updates ELO of last model)
  :bench <prompt>  run prompt across all discovered models (timing only)
  :reset           clear conversation memory (history)
  :history         show how many turns are remembered
  :clear           clear screen
  :quit            exit
"""


def color_face(state: str) -> str:
    if state == "happy":
        return f"{C_GREEN}(^_^){C_RESET}"
    if state == "thinking":
        return f"{C_YELLOW}(o_O){C_RESET}"
    if state == "confused":
        return f"{C_MAGENTA}(?_?){C_RESET}"
    if state == "unhappy":
        return f"{C_RED}(>_<){C_RESET}"
    if state == "tired":
        return f"{C_DIM}(-_-) zZ{C_RESET}"
    return "(-_-)"


def print_models(active):
    models = discover_models()
    print(f"\n{C_BOLD}Models discovered in {MODELS_DIR}:{C_RESET}")
    if not models:
        print(f"  {C_YELLOW}(none — drop *.gguf files into {MODELS_DIR}){C_RESET}\n")
        return
    for mid, cfg in models.items():
        active_marker = f"{C_CYAN}<- active{C_RESET}" if mid == active else ""
        print(f"  {mid:40s}  {cfg['label']}  {active_marker}")
    print()


def run_benchmark(registry: LLMRegistry, prompt_text: str):
    ids = available_model_ids()
    if not ids:
        print(f"{C_RED}No models found in {MODELS_DIR}. Drop *.gguf files there.{C_RESET}")
        return
    mode = detect_agent_mode(prompt_text)
    prompt = build_prompt(prompt_text, mode)
    print(f"\n{C_BOLD}Benchmark across {len(ids)} model(s):{C_RESET}\n")
    for mid in ids:
        cfg = get_model(mid)
        print(f"  {C_CYAN}{mid}{C_RESET} ({cfg['label']})")
        try:
            t0 = time.time()
            reply = registry.generate(mid, prompt)
            elapsed = time.time() - t0
            tokens = max(1, len(reply.split()))
            tps = tokens / elapsed if elapsed > 0 else 0
            print(f"    {C_DIM}elapsed:{C_RESET} {elapsed:6.2f}s   "
                  f"{C_DIM}~tokens:{C_RESET} {tokens:4d}   "
                  f"{C_DIM}~tok/s:{C_RESET} {tps:5.2f}")
            head = reply.strip().splitlines()[0][:80] if reply.strip() else "(empty)"
            print(f"    {C_DIM}preview:{C_RESET} {head}\n")
        except ModelNotAvailable as e:
            print(f"    {C_RED}skip:{C_RESET} {e}\n")
        except Exception as e:
            print(f"    {C_RED}error:{C_RESET} {e}\n")


def run_cli(initial_model: str | None):
    registry = LLMRegistry()
    elo_store = EloStore(ELO_STATE_PATH)
    router = Router(RAG_EMBEDDINGS_PATH, RAG_META_PATH, BERT_MODEL_NAME)
    active = initial_model or default_model_id()
    history = []  # list of {"role": "user"|"assistant", "content": str}
    auto_routing = True
    last_model_id = None
    last_target_elo = None

    print(f"{C_CYAN}{BANNER}{C_RESET}")
    if active is None:
        print(f"{C_YELLOW}No models found in {MODELS_DIR}.{C_RESET}")
        print(f"{C_DIM}Drop *.gguf files into that directory, then use ':models'.{C_RESET}")
    else:
        cfg = get_model(active)
        if cfg is None:
            print(f"{C_YELLOW}Requested model '{active}' not found on disk.{C_RESET}")
        else:
            print(f"Active model: {C_BOLD}{active}{C_RESET}  ({cfg['label']})")
    print(HELP)

    while True:
        try:
            user = input(f"{C_BOLD}you{C_RESET} > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{C_DIM}bye!{C_RESET}")
            return

        if not user:
            continue

        if user in (":quit", ":exit"):
            print(f"{C_DIM}bye!{C_RESET}")
            return
        if user == ":help":
            print(HELP)
            continue
        if user == ":clear":
            print("\033[2J\033[H", end="")
            continue
        if user == ":models":
            print_models(active)
            continue
        if user == ":reset":
            history.clear()
            print(f"{C_DIM}conversation memory cleared{C_RESET}")
            continue
        if user == ":history":
            turns = len([h for h in history if h["role"] == "user"])
            print(f"{C_DIM}remembering {turns} turn(s) "
                  f"(cap {MAX_HISTORY_TURNS}){C_RESET}")
            continue
        if user.startswith(":use "):
            target = user[len(":use "):].strip()
            if get_model(target) is None:
                print(f"{C_RED}unknown model_id:{C_RESET} {target}")
                print(f"{C_DIM}use ':models' to see what's discovered{C_RESET}")
                continue
            active = target
            auto_routing = False
            print(f"active model: {C_BOLD}{active}{C_RESET}  "
                  f"({get_model(active)['label']})  "
                  f"{C_DIM}[auto routing OFF]{C_RESET}")
            continue
        if user.startswith(":auto"):
            arg = user[len(":auto"):].strip().lower()
            if arg in ("", "status"):
                state = "ON" if auto_routing else "OFF"
                src = "BERT+RAG" if router.using_bert else "keyword fallback"
                print(f"auto routing: {C_BOLD}{state}{C_RESET}  "
                      f"({src}){' — ' + router.load_error if router.load_error else ''}")
            elif arg == "on":
                auto_routing = True
                print(f"{C_DIM}auto routing ON{C_RESET}")
            elif arg == "off":
                auto_routing = False
                print(f"{C_DIM}auto routing OFF — using {active}{C_RESET}")
            else:
                print(f"{C_RED}usage:{C_RESET} :auto on | :auto off | :auto status")
            continue
        if user.startswith(":route "):
            text = user[len(":route "):].strip()
            info = router.route(text, available_model_ids(), elo_store)
            chosen = info.get("chosen_model_id") or "(none)"
            print(f"  LC rating  ≈ {C_BOLD}{info['predicted_lc_rating']}{C_RESET}")
            print(f"  target ELO = {C_BOLD}{info['target_elo']}{C_RESET}")
            print(f"  source     = {info['source']}")
            print(f"  chosen     = {C_CYAN}{chosen}{C_RESET}  (ELO {info.get('model_elo')})")
            for n in info.get("neighbors") or []:
                print(f"    ~ {n['title']:60s}  rating {n['rating']:.0f}  sim {n['similarity']:.2f}")
            continue
        if user == ":elo":
            ids = available_model_ids()
            if not ids:
                print(f"{C_YELLOW}no models discovered{C_RESET}")
            for mid in ids:
                print(f"  {mid:40s}  ELO {elo_store.get(mid)}")
            continue
        if user.startswith(":vote"):
            arg = user[len(":vote"):].strip().lower()
            if arg not in ("up", "down"):
                print(f"{C_RED}usage:{C_RESET} :vote up | :vote down")
                continue
            if last_model_id is None:
                print(f"{C_YELLOW}no previous reply to vote on{C_RESET}")
                continue
            target = last_target_elo if last_target_elo is not None else elo_store.get(last_model_id)
            new_elo = update_elo(elo_store, last_model_id, int(target), arg)
            print(f"  {last_model_id}: ELO -> {C_BOLD}{new_elo}{C_RESET}  "
                  f"({arg} vs target {target})")
            continue
        if user.startswith(":bench "):
            run_benchmark(registry, user[len(":bench "):].strip())
            continue

        classification = classify_query(user, history=history)

        if classification == "rude":
            print(f"{color_face('unhappy')} {C_RED}{RUDE_REPLY}{C_RESET}\n")
            continue
        if classification == "greeting":
            print(f"{color_face('happy')} {C_GREEN}{GREETING_REPLY}{C_RESET}\n")
            continue
        if classification == "identity":
            print(f"{color_face('happy')} {C_GREEN}{IDENTITY_REPLY}{C_RESET}\n")
            continue
        if classification == "too_hard":
            print(f"{color_face('tired')} {C_DIM}{GIVEUP_REPLY}{C_RESET}\n")
            continue
        if classification == "non_coding":
            print(f"{color_face('confused')} {C_MAGENTA}{NON_CODING_REPLY}{C_RESET}\n")
            continue

        if active is None and not auto_routing:
            print(f"{color_face('unhappy')} {C_RED}No model available. Place a *.gguf in {MODELS_DIR}.{C_RESET}\n")
            continue

        # Routing decision
        if auto_routing:
            ids = available_model_ids()
            info = router.route(user, ids, elo_store)
            chosen = info.get("chosen_model_id") or active
            last_target_elo = info["target_elo"]
            if chosen is None:
                print(f"{color_face('unhappy')} {C_RED}No model available.{C_RESET}\n")
                continue
            print(f"{C_DIM}[router] LC≈{info['predicted_lc_rating']}, "
                  f"target ELO {info['target_elo']}, "
                  f"picked {chosen} (ELO {info.get('model_elo')}, src {info['source']}){C_RESET}")
        else:
            chosen = active
            last_target_elo = elo_store.get(chosen) if chosen else None

        last_model_id = chosen

        mode = detect_agent_mode(user)
        prompt = build_prompt(user, mode, history=history,
                              max_history_turns=MAX_HISTORY_TURNS)
        turns_used = min(len([h for h in history if h["role"] == "user"]),
                         MAX_HISTORY_TURNS)
        print(f"{color_face('thinking')} {C_DIM}thinking ({mode}, "
              f"remembering {turns_used} turn(s))...{C_RESET}")
        try:
            t0 = time.time()
            reply = registry.generate(chosen, prompt)
            elapsed = time.time() - t0
            print(f"{color_face('happy')} {C_GREEN}CodePi{C_RESET} "
                  f"{C_DIM}[{chosen} | {elapsed:.2f}s | "
                  f"vote with :vote up|down]{C_RESET}:")
            print(reply + "\n")
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": reply})
        except ModelNotAvailable as e:
            print(f"{color_face('unhappy')} {C_RED}{e}{C_RESET}\n")
        except Exception as e:
            print(f"{color_face('unhappy')} {C_RED}LLM error: {e}{C_RESET}\n")
