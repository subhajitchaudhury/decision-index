#!/usr/bin/env python3
"""Minimal, single-purpose decider for IBM's Granite Decision LoRA on the JEV Decision Index.

Scope: ONLY the format the adapter was trained on (ibm-granite/granitelib-core-r1.0 model card) --
one user turn, no system message, 'A: desc' lettered options, single-token LETTER labels (A-Z then
digits / letter-pairs for wide questions), one constrained forward pass per question. No layout
switches, no engine search -- those live in the general harness (gg_clm/eikos_lora/). The core
readout, per-type reduction and run loop match the general driver, so output is byte-compatible
with the kit scorer.

  python decider_granite_decision_lora.py --base-url http://127.0.0.1:8231 --model decision-lora \\
      --tokenizer <granite-4.1-30b dir> --suite <suite-0.2 dir> --out <run>/results.jsonl \\
      [--conc 48] [--maxlen 32768]
"""
import argparse, asyncio, gzip, json, math, os, time

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True, help="vLLM OpenAI endpoint, e.g. http://127.0.0.1:8231")
ap.add_argument("--model", default="decision-lora", help="served adapter name to request")
ap.add_argument("--tokenizer", required=True, help="base granite-4.1-30b dir (for labels + length)")
ap.add_argument("--suite", required=True, help="frozen suite-0.2 dir (selected-rows + added-rows)")
ap.add_argument("--out", required=True, help="results.jsonl to append")
ap.add_argument("--conc", type=int, default=48)
ap.add_argument("--maxlen", type=int, default=32768)
A = ap.parse_args()
BASE, SERVED, SUITE, OUT, CONC, MAXLEN = A.base_url, A.model, A.suite, A.out, A.conc, A.maxlen

from transformers import AutoTokenizer  # noqa: E402
_tok = AutoTokenizer.from_pretrained(A.tokenizer, trust_remote_code=True)

# Label pool = card scheme: letters A..Z first, then single-token digits, then letter-pairs AA,AB,...
# ("single-token labels: A-Z, then digits / letter-pairs for wide questions"). Letters first so the
# overwhelmingly common narrow questions read exactly as trained; the tail only covers high-
# cardinality families (BANKING77 77, CLINC150 151). Keep only one-token labels for this tokenizer.
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LABELS, LABEL_IDS = [], {}
for _lab in list(_LETTERS) + [str(i) for i in range(10)] + [a + b for a in _LETTERS for b in _LETTERS]:
    _ids = _tok(_lab, add_special_tokens=False)["input_ids"]
    if len(_ids) == 1 and _lab not in LABEL_IDS:
        LABELS.append(_lab); LABEL_IDS[_lab] = _ids[0]


class Capacity(Exception):
    pass


def _text(x):
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, separators=(",", ":"))


def _labels_for(n):
    if n > len(LABELS):
        raise Capacity(f"{n} options per choice exceeds single-token labels (pool {len(LABELS)})")
    return LABELS[:n]


def render_prompt(state, instructions, labels, descriptions):
    """IBM decision-LoRA card format, byte-for-byte: ONE user turn, NO system message, options as
    'A: desc' lines, closed with 'Reply with exactly one option label.' and add_generation_prompt
    so the next token IS the option label."""
    lines = ["State:", _text(state) if state not in ("", None, {}, []) else "(empty)",
             "", "Question:", _text(instructions), "", "Options:"]
    for lab, d in zip(labels, descriptions):
        lines.append(f"{lab}: {_text(d) if d not in (None, '') else lab}")
    lines += ["", "Reply with exactly one option label."]
    msgs = [{"role": "user", "content": "\n".join(lines)}]
    try:
        return _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


_REQ_SEM = None


async def _readout_probs(client, state, instructions, descriptions):
    """The one call to vLLM: single constrained next-token readout -> softmax over the option
    labels (same order). Read the label logprobs at the first assistant position."""
    n = len(descriptions)
    labels = _labels_for(n)
    prompt = render_prompt(state, instructions, labels, descriptions)
    ntok = len(_tok(prompt, add_special_tokens=False)["input_ids"])
    if ntok > MAXLEN - 4:
        raise Capacity(f"maximum context length: {ntok} tokens")
    # top-k over vocab can rank non-label tokens between labels, so request n+15 (vLLM caps at 20);
    # allowed_token_ids restricts generation to the labels.
    body = {"model": SERVED, "prompt": prompt, "max_tokens": 1, "temperature": 0,
            "logprobs": min(n + 15, 20), "allowed_token_ids": [LABEL_IDS[l] for l in labels]}
    async with _REQ_SEM:
        r = await client.post(BASE + "/v1/completions", json=body)
    r.raise_for_status()
    top = r.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    out = []
    for lab in labels:
        if lab in top:
            out.append(top[lab])
        else:
            v = next((val for k, val in top.items() if k.strip() == lab), None)
            out.append(v if v is not None else -1e9)
    m = max(out)
    ex = [math.exp(x - m) for x in out]      # temperature 1.0: only rescales probs, never the argmax
    z = sum(ex)
    return [e / z for e in ex]


async def _answer_one(client, state, q):
    """One question -> its kit answer (choice / noul / score). Independent -> run concurrently."""
    t = q.get("type")
    if t == "choice":
        crit = q.get("criteria")
        if not isinstance(crit, dict) or len(crit) < 2:
            raise Capacity("a choice needs at least two options")
        keys = list(crit)
        probs = await _readout_probs(client, state, q.get("instructions"), [crit[k] for k in keys])
        j = max(range(len(keys)), key=lambda i: probs[i])
        return {"type": "choice", "choice": keys[j],
                "probabilities": {k: float(p) for k, p in zip(keys, probs)}}
    if t == "noul":
        crit = q.get("criteria") if isinstance(q.get("criteria"), dict) else {}
        descs = [crit.get("false") or "No / false", crit.get("true") or "Yes / true"]
        probs = await _readout_probs(client, state, q.get("instructions"), descs)
        return {"type": "noul", "noul": float(probs[1])}
    if t == "score":
        levels = q.get("criteria")
        if not isinstance(levels, list) or len(levels) < 2:
            raise Capacity("a score takes 2 to 10 levels")
        probs = await _readout_probs(client, state, q.get("instructions"), levels)
        return {"type": "score", "score": float(sum(i * p for i, p in enumerate(probs))),
                "probabilities": {str(i): float(p) for i, p in enumerate(probs)}}
    raise Capacity("Unsupported question type " + str(t))


async def answer_row(client, state, questions):
    qids = list(questions)
    results = await asyncio.gather(*[_answer_one(client, state, questions[qid]) for qid in qids])
    return dict(zip(qids, results))


def iter_rows():
    for fn in ("selected-rows.jsonl.gz", "added-rows.jsonl.gz"):
        with gzip.open(os.path.join(SUITE, fn), "rt") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


async def main():
    import httpx
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):                 # resume: skip run_ids already answered ok
        for l in open(OUT):
            try:
                r = json.loads(l)
                if r.get("status") != "error":
                    done.add(r["run_id"])
            except Exception:
                pass
    rows = [r for r in iter_rows() if r["_evaluation"]["run_id"] not in done]
    total = len(rows)
    print(f"[decider] resume {len(done)} done, {total} to do | req-conc={CONC} "
          f"labels={len(LABELS)} layout=granitedec", flush=True)
    outf = open(OUT, "a")
    lock = asyncio.Lock()
    global _REQ_SEM
    _REQ_SEM = asyncio.Semaphore(CONC)
    row_sem = asyncio.Semaphore(CONC * 4)
    counts = {}; n = 0; start = time.time()
    limits = httpx.Limits(max_connections=CONC * 2 + 16, max_keepalive_connections=CONC * 2)
    async with httpx.AsyncClient(timeout=180, limits=limits) as client:
        async def handle(row):
            nonlocal n
            e = row["_evaluation"]; rec = {**e, "engine": "decider-chat"}
            try:
                async with row_sem:
                    ans = await answer_row(client, row["state"], row["questions"])
                rec.update(status="ok", response={"model": SERVED, "answers": ans},
                           total_wall_ms=50.0, model_request_wall_ms=50.0)
            except Capacity as ex:
                msg = str(ex)
                marker = any(s in msg for s in ("maximum context length", "options per choice",
                                                "a choice needs", "a score takes"))
                rec.update(status="unsupported" if marker else "error", error=msg,
                           total_wall_ms=50.0, model_request_wall_ms=50.0)
            except Exception as ex:  # noqa: BLE001
                rec.update(status="error", error=f"{type(ex).__name__}: {ex}"[:200],
                           total_wall_ms=50.0, model_request_wall_ms=50.0)
            async with lock:
                outf.write(json.dumps(rec) + "\n"); outf.flush()
                counts[rec["status"]] = counts.get(rec["status"], 0) + 1
                n += 1
                if n % 500 == 0:
                    el = time.time() - start
                    print(f"[decider] {n}/{total} ({n/el:.1f} rows/s) {counts} "
                          f"eta={(total-n)/max(n/el,0.1)/60:.0f}min", flush=True)
        await asyncio.gather(*[handle(r) for r in rows])
    outf.close()
    print(f"[decider] DONE {n}/{total} in {(time.time()-start)/60:.1f}min {counts}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
