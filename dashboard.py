"""
Streamlit dashboard for P05 (Ollama version)
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
from gateway import LLMOpsGateway

st.set_page_config(page_title="P05 LLMOps Dashboard (Ollama)", layout="wide")

st.title("P05 — LLMOps Unit Economics Dashboard")
st.caption("Ollama • Caching • Cost attribution • Circuit breaker")

if "gateway" not in st.session_state:
    with st.spinner("Initializing Ollama gateway..."):
        st.session_state.gateway = LLMOpsGateway()
    st.success("Gateway ready")

gateway = st.session_state.gateway

col1, col2 = st.columns(2)

with col1:
    st.subheader("Run a request")
    team = st.selectbox("Team", ["engineering", "data-science", "product", "support"])
    workload = st.selectbox("Workload", ["research", "coding", "chat", "batch"])
    prompt = st.text_area("Prompt", "What is the caching policy?")
    if st.button("Send through Gateway"):
        with st.spinner("Calling model..."):
            result = gateway.call(prompt, team=team, workload=workload)
        if result.get("stopped"):
            st.error(result["text"])
        else:
            st.success("Success")
            st.write(result["text"])
            st.json({
                "cost_usd": result["cost_usd"],
                "cached_tokens": result["cached_tokens"],
                "cache_hit": result["cache_hit"],
            })

with col2:
    st.subheader("Simulate runaway loop")
    if st.button("Start runaway loop (will be stopped)"):
        depth = 1
        status = st.empty()
        while depth <= 15:
            result = gateway.call(
                f"Keep going. Step {depth}",
                team="engineering",
                workload="research",
                depth=depth,
            )
            if result.get("stopped"):
                status.error(f"Stopped at depth {depth}: {result['reason']}")
                break
            status.info(f"Step {depth} completed. Team spend: ${result['team_spend_so_far']:.6f}")
            depth += 1

st.divider()
st.subheader("Live Metrics")

d = gateway.get_dashboard()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Cache-Hit Rate", f"{d['cache_hit_rate_pct']}%")
m2.metric("Actual Cost", f"${d['total_cost_usd']:.6f}")
m3.metric("Savings", f"${d['savings_usd']:.6f}")
m4.metric("Cap Trips", d["cap_trips"])

if gateway.stats.requests:
    rows = []
    for r in gateway.stats.requests[-20:]:
        rows.append({
            "Time": r.timestamp[11:19],
            "Team": r.team,
            "Workload": r.workload,
            "Cache Hit": r.cache_hit,
            "Cost": f"${r.cost_usd:.6f}",
            "Stopped": r.stopped_by_breaker,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
