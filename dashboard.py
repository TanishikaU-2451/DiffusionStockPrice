from __future__ import annotations

import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from diffusion_finance.service import ModelService


st.set_page_config(
    page_title="Financial Anomaly Dashboard",
    layout="wide",
)
st.title("MSFT Anomaly Detection and Pattern Labeling")

service = ModelService()
pipeline = service.ensure_ready()
results = pd.DataFrame(pipeline.simulate_stream())
threshold = pipeline.state.threshold

st.sidebar.header("Controls")
replay_delay = st.sidebar.slider(
    "Replay delay (ms)",
    min_value=0,
    max_value=1000,
    value=150,
    step=50,
)
limit = st.sidebar.slider(
    "Windows to visualize",
    min_value=25,
    max_value=len(results),
    value=min(200, len(results)),
)
show_labels = st.sidebar.checkbox("Show labels table", value=True)
start_replay = st.sidebar.button("Start real-time replay")


def render_dashboard(frame: pd.DataFrame) -> None:
    anomaly_frame = frame[frame["anomaly_flag"]]

    price_chart = go.Figure()
    price_chart.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["close"],
            mode="lines",
            name="Close",
            line={"color": "#1f77b4", "width": 2},
        )
    )
    price_chart.add_trace(
        go.Scatter(
            x=anomaly_frame["timestamp"],
            y=anomaly_frame["close"],
            mode="markers",
            name="Anomaly",
            marker={"color": "#d62728", "size": 9},
            text=anomaly_frame["label"],
            hovertemplate="%{x}<br>Close=%{y:.2f}<br>%{text}<extra></extra>",
        )
    )
    price_chart.update_layout(
        title="Stock Price With Detected Anomalies",
        xaxis_title="Time",
        yaxis_title="Close Price",
        height=420,
    )

    score_chart = go.Figure()
    score_chart.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=frame["anomaly_score"],
            mode="lines",
            name="Anomaly score",
            line={"color": "#ff7f0e", "width": 2},
        )
    )
    score_chart.add_trace(
        go.Scatter(
            x=frame["timestamp"],
            y=[threshold] * len(frame),
            mode="lines",
            name="Dynamic threshold",
            line={"color": "#2ca02c", "dash": "dash"},
        )
    )
    score_chart.update_layout(
        title="Anomaly Score Trend",
        xaxis_title="Time",
        yaxis_title="Score",
        height=340,
    )

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(price_chart, use_container_width=True)
        st.plotly_chart(score_chart, use_container_width=True)
    with right:
        st.metric("Windows analyzed", len(frame))
        st.metric("Anomalies", int(frame["anomaly_flag"].sum()))
        st.metric("Latest label", frame["label"].iloc[-1] if not frame.empty else "N/A")
        latest_score = f"{frame['anomaly_score'].iloc[-1]:.4f}" if not frame.empty else "N/A"
        st.metric("Latest score", latest_score)
        if show_labels:
            st.dataframe(
                frame[["timestamp", "close", "label", "anomaly_score", "anomaly_flag"]].tail(20),
                use_container_width=True,
                hide_index=True,
            )


if start_replay:
    placeholder = st.empty()
    replay_frame = results.head(limit).copy()
    for end_index in range(1, len(replay_frame) + 1):
        with placeholder.container():
            render_dashboard(replay_frame.iloc[:end_index])
        time.sleep(replay_delay / 1000)
else:
    render_dashboard(results.head(limit))
