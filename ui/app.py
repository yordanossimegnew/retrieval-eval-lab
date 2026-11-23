"""Streamlit UI for the Retrieval Evaluation Lab.

    uvicorn main:app            # terminal 1
    streamlit run ui/app.py     # terminal 2
"""

from __future__ import annotations

import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("AGENT_API_URL", "http://localhost:8000")
SERIES = "#2a78d6"          # reference palette, categorical slot 1 (single-series charts)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"
CHUNKINGS = {"300 / 50": (300, 50), "800 / 100": (800, 100), "1200 / 150": (1200, 150), "2400 / 300": (2400, 300)}

st.set_page_config(page_title="Retrieval Eval Lab", page_icon="🧪", layout="wide")
st.markdown("""<style>
.chunk {border:1px solid #e5e7eb;border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem}
.chunk.rel {border-color:#16a34a;background:#f0fdf4}
.tag {font-size:0.75rem;font-weight:700;margin-right:6px}
</style>""", unsafe_allow_html=True)


def api(method, path, **kw):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=600, **kw)
        body = r.json()
        return body if r.ok else {"_error": body.get("detail", r.text)}
    except requests.RequestException as exc:
        return {"_error": str(exc)}


def done(message: str) -> None:
    """Show a success message after a rerun, so every tab sees the new data."""
    st.session_state.flash = message
    st.rerun()


def is_relevant(chunk: dict, relevant: list[dict]) -> bool:
    text = " ".join(chunk["text"].lower().split())
    return any(chunk["source"] == u["doc_key"] and (not u.get("evidence") or " ".join(u["evidence"].lower().split()) in text)
               for u in relevant)


def bar_chart(df: pd.DataFrame, field: str, title: str):
    """Single-measure horizontal bars, sorted, values labelled at the bar end."""
    base = alt.Chart(df).encode(
        y=alt.Y("config:N", sort="-x", title=None,
                axis=alt.Axis(labelLimit=320, labelColor=MUTED, labelOverlap=False)),
        x=alt.X(f"{field}:Q", title=title, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(grid=True, gridColor=GRID, labelColor=MUTED, titleColor=MUTED, format=".1f")),
        tooltip=[alt.Tooltip("config:N"), alt.Tooltip("Recall@k:Q", format=".1%"), alt.Tooltip("Precision@k:Q", format=".1%"),
                 alt.Tooltip("MRR:Q", format=".3f"), alt.Tooltip("nDCG@k:Q", format=".3f"),
                 alt.Tooltip("p50 ms:Q", format=".1f"), alt.Tooltip("p95 ms:Q", format=".1f")],
    )
    bars = base.mark_bar(color=SERIES, cornerRadiusEnd=4, height=14)
    labels = base.mark_text(align="left", dx=4, color=INK, fontSize=11).encode(text=alt.Text(f"{field}:Q", format=".2f"))
    return (bars + labels).properties(height=max(120, 30 * len(df))).configure_view(strokeWidth=0)


def tradeoff_chart(df: pd.DataFrame):
    """Quality vs speed: each config is a point (one series, so one color); hover for the name."""
    return alt.Chart(df).mark_circle(size=110, color=SERIES, opacity=0.9, stroke="#fcfcfb", strokeWidth=2).encode(
        x=alt.X("p50 ms:Q", title="Median latency per query (ms)",
                axis=alt.Axis(gridColor=GRID, labelColor=MUTED, titleColor=MUTED)),
        y=alt.Y("nDCG@k:Q", title="nDCG@k", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(gridColor=GRID, labelColor=MUTED, titleColor=MUTED)),
        tooltip=["config:N", alt.Tooltip("nDCG@k:Q", format=".3f"), alt.Tooltip("p50 ms:Q", format=".1f"),
                 alt.Tooltip("$/1k queries:Q", format="$.4f")],
    ).properties(height=300).configure_view(strokeWidth=0)


# ----------------------------------------------------------------------------- sidebar
health = api("GET", "/health")
ok = "_error" not in health
with st.sidebar:
    st.title("🧪 Retrieval Eval Lab")
    st.caption("Measuring the Document QA retriever")
    if ok:
        st.markdown(f"**Embeddings:** `{health['embeddings']}`  \n**Reranker:** `{health['reranker']}`  \n"
                    f"**Claude:** `{health['model']}`")
    else:
        st.error(f"API not reachable at {API_URL}. Start it with `uvicorn main:app`.", icon="🚫")
    corpora = api("GET", "/corpora") if ok else []
    qsets = api("GET", "/question-sets") if ok else []
    corpora = corpora if isinstance(corpora, list) else []
    qsets = qsets if isinstance(qsets, list) else []
    st.markdown("### Corpora")
    for c in corpora:
        st.markdown(f"- **{c['name']}** · {c['documents']} docs")
    st.markdown("### Question sets")
    for q in qsets:
        status = q["meta"].get("status", "ready")
        st.markdown(f"- **{q['name']}** · {q['questions']} questions" + ("" if status == "ready" else f" · _{status}_"))

if "flash" in st.session_state:
    st.success(st.session_state.pop("flash"))
tab_data, tab_run, tab_results = st.tabs(["① Data", "② Experiment", "③ Results"])

# ----------------------------------------------------------------------------- data
with tab_data:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Your documents")
        name = st.text_input("Corpus name", "My documents")
        files = st.file_uploader("PDF, DOCX, Markdown, text or HTML", accept_multiple_files=True,
                                 type=["pdf", "docx", "md", "txt", "html"])
        if st.button("Create corpus", disabled=not (ok and files)):
            res = api("POST", "/corpora", data={"name": name}, files=[("files", (f.name, f.getvalue())) for f in files])
            st.error(res["_error"]) if "_error" in res else done(f"Corpus “{res['name']}” created")
        st.markdown("#### Synthetic questions (Claude)")
        if corpora:
            corpus = st.selectbox("Corpus", corpora, format_func=lambda c: c["name"])
            passages = st.slider("Passages to sample", 5, 200, 30)
            per = st.slider("Questions per passage", 1, 5, 2)
            if st.button("Generate questions"):
                res = api("POST", "/question-sets/synthetic", json={"corpus_id": corpus["id"], "passages": passages,
                                                                    "per_passage": per, "name": f"Synthetic · {corpus['name']}"})
                st.error(res["_error"]) if "_error" in res else done("Generating questions… the set appears in the sidebar when ready.")
    with c2:
        st.markdown("#### Public benchmark: SciFact (BEIR)")
        st.caption("Scientific claims paired with the abstracts that support or refute them.")
        mq = st.slider("Test queries", 10, 300, 50)
        md = st.slider("Abstracts (relevant + distractors)", 100, 2000, 400, step=100)
        if st.button("Load SciFact", disabled=not ok):
            with st.spinner("Downloading SciFact from Hugging Face…"):
                res = api("POST", "/corpora/benchmark", json={"max_queries": mq, "max_docs": md})
            st.error(res["_error"]) if "_error" in res else done(f"Loaded {res['corpus']['documents']} SciFact abstracts and "
                                                                 f"{res['question_set']['questions']} test queries")
    if qsets:
        st.markdown("#### Browse a question set")
        qs = st.selectbox("Question set", qsets, format_func=lambda q: f"{q['name']} ({q['questions']})", key="browse")
        detail = api("GET", f"/question-sets/{qs['id']}")
        if "items" in detail:
            rows = [{"question": i["question"], "relevant document": ", ".join(u["doc_key"] for u in i["relevant"]),
                     "evidence": "; ".join(u["evidence"] or "(whole document)" for u in i["relevant"])} for i in detail["items"]]
            st.dataframe(rows, hide_index=True, width="stretch", height=260)
            if detail["meta"].get("dropped"):
                st.caption(f"Quality filter dropped {detail['meta']['dropped']} generated questions "
                           "(evidence not an exact quote, duplicates, malformed).")

# ----------------------------------------------------------------------------- experiment
with tab_run:
    ready = [q for q in qsets if q["questions"] and q["meta"].get("status", "ready") == "ready"]
    if not ready:
        st.info("Create a question set first (synthetic or SciFact).")
    else:
        qs = st.selectbox("Question set", ready, format_func=lambda q: f"{q['name']} ({q['questions']} questions)")
        c1, c2, c3 = st.columns(3)
        retrievers = c1.multiselect("Retrievers", ["vector", "keyword", "hybrid"], ["vector", "keyword", "hybrid"])
        chunkings = c1.multiselect("Chunk size / overlap (chars)", list(CHUNKINGS), ["800 / 100", "2400 / 300"])
        top_k = c2.multiselect("Top k", [3, 5, 10], [5])
        rerank = c2.multiselect("Reranking (Voyage)", ["off", "on"], ["off", "on"])
        transforms = c3.multiselect("Query transform (Claude)", ["none", "rewrite", "hyde"], ["none"])
        n = len(retrievers) * len(chunkings) * len(top_k) * len(rerank) * len(transforms)
        st.markdown(f"**{n} configurations** × {qs['questions']} questions")
        name = st.text_input("Run name", "Retriever comparison")
        if st.button("Start run", type="primary", disabled=n == 0 or n > 48):
            res = api("POST", "/runs", json={"question_set_id": qs["id"], "name": name, "grid": {
                "retrievers": retrievers, "chunk_sizes": [CHUNKINGS[c] for c in chunkings], "top_k": top_k,
                "rerank": [r == "on" for r in rerank], "query_transforms": transforms}})
            if "_error" in res:
                st.error(res["_error"])
            else:
                st.session_state.run_id = res["id"]
                st.success("Run started. Watch it in ③ Results.")


# ----------------------------------------------------------------------------- results
@st.fragment(run_every=1.5)
def results_panel():
    runs = api("GET", "/runs")
    if not isinstance(runs, list) or not runs:
        st.info("No runs yet.")
        return
    ids = [r["id"] for r in runs]
    default = ids.index(st.session_state["run_id"]) if st.session_state.get("run_id") in ids else 0
    run_id = st.selectbox("Run", ids, index=default, key="run_pick",
                          format_func=lambda i: next(f"{r['name']} · {r['status']}" for r in runs if r["id"] == i))
    run = api("GET", f"/runs/{run_id}")
    prog = run["progress"]
    if run["status"] in ("queued", "running"):
        st.progress(prog["done"] / max(1, prog["total"]),
                    text=f"{prog.get('stage', '')}: {prog['done']}/{prog['total']} configurations")
    if run["status"] == "failed":
        st.error(run["error"])
    if not run["results"]:
        return

    df = pd.DataFrame([{
        "config": r["config_key"], "Recall@k": r["metrics"]["recall"], "Precision@k": r["metrics"]["precision"],
        "MRR": r["metrics"]["mrr"], "nDCG@k": r["metrics"]["ndcg"], "p50 ms": r["latency"]["p50_ms"],
        "p95 ms": r["latency"]["p95_ms"], "$/1k queries": r["cost"]["usd_per_1k_queries"]} for r in run["results"]])
    best = df.iloc[0]
    c = st.columns(4)
    c[0].metric("Best config", best["config"].split(" · ")[0], help=best["config"])
    c[1].metric("Its nDCG@k", f"{best['nDCG@k']:.3f}")
    c[2].metric("Its Recall@k", f"{best['Recall@k']:.0%}")
    c[3].metric("Configs evaluated", f"{prog['done']}/{prog['total']}")
    st.caption(f"Best: **{best['config']}**")

    st.markdown("#### Leaderboard")
    st.dataframe(df.style.format({"Recall@k": "{:.1%}", "Precision@k": "{:.1%}", "MRR": "{:.3f}", "nDCG@k": "{:.3f}",
                                  "p50 ms": "{:.1f}", "p95 ms": "{:.1f}", "$/1k queries": "${:.4f}"}),
                 hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("#### nDCG@k by configuration")
        st.altair_chart(bar_chart(df, "nDCG@k", "nDCG@k"), width="stretch")
    with right:
        st.markdown("#### Quality vs. latency")
        st.altair_chart(tradeoff_chart(df), width="stretch")

    st.markdown("#### Per-question drill-down")
    d1, d2 = st.columns([3, 1])
    cfg = d1.selectbox("Configuration", list(df["config"]), key="drill_cfg")
    misses = d2.toggle("Only questions it missed", value=False)
    rows = api("GET", f"/runs/{run_id}/queries", params={"config_key": cfg, "only_misses": misses})
    for row in (rows if isinstance(rows, list) else [])[:40]:
        m = row["metrics"]
        icon = "✅" if m["hit"] else "❌"
        with st.expander(f"{icon} {row['question']}  ·  recall {m['recall']:.0%} · MRR {m['mrr']:.2f}"):
            if row["query_used"] != row["question"]:
                st.caption(f"Query sent: {row['query_used']}")
            ev = [u.get("evidence") for u in row["relevant"] if u.get("evidence")]
            st.caption("Relevant: " + ", ".join(u["doc_key"] for u in row["relevant"]) + (f" · evidence: “{ev[0]}”" if ev else ""))
            for rank, ch in enumerate(row["retrieved"], 1):
                rel = is_relevant(ch, row["relevant"])
                st.markdown(f'<div class="chunk {"rel" if rel else ""}"><span class="tag">#{rank} {"✓ relevant" if rel else ""}</span>'
                            f'<b>{ch["title"]}</b> <span style="color:#64748b">· {ch["source"]} · score {ch["score"]}</span><br>'
                            f'{ch["text"][:300]}…</div>', unsafe_allow_html=True)


with tab_results:
    results_panel()
