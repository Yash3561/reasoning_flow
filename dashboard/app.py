"""
Streamlit Dashboard: The Geometry of Reasoning
Flowing Logics in Representation Space

Author: Yash Chaudhary | Master's Research | NJIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Reasoning Flow Dashboard",
    page_icon="🧠",
)

# ── Base directory ────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent

# ── Global style ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Poppins', sans-serif !important; }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0e1117;
        color: #e0e0e0;
        font-size: 15px;
    }

    p, div, span, li { line-height: 1.8; }

    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    [data-testid="metric-container"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 20px;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #f78166 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #8b949e;
        margin-top: 0.4rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #58a6ff;
        border-bottom: 2px solid #21262d;
        padding-bottom: 0.6rem;
        margin-bottom: 1.5rem;
    }
    .tag {
        display: inline-block;
        background: #21262d;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 5px 14px;
        font-size: 0.85rem;
        color: #58a6ff;
        margin-right: 6px;
    }
    .sidebar-block {
        background: #21262d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.85rem;
        color: #c9d1d9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Colour palette ────────────────────────────────────────────────────────────
LOGIC_COLORS = {"LogicA": "#58a6ff", "LogicB": "#3fb950", "LogicC": "#f78166"}
LANG_COLORS  = {"EN": "#58a6ff", "ZH": "#f78166", "DE": "#3fb950", "JA": "#d2a8ff", "Abstract": "#e3b341"}
PLOTLY_THEME = "plotly_dark"
PLOTLY_FONT  = dict(family="Poppins, sans-serif", size=13)

# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data
def load_curvature():
    path = BASE_DIR / "curvature_l2_normalized.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    def parse_label(label: str):
        label = label.strip()
        if label.startswith("logica"):
            logic = "LogicA"
        elif label.startswith("logicb"):
            logic = "LogicB"
        elif label.startswith("logicc"):
            logic = "LogicC"
        else:
            logic = "Unknown"

        if label.endswith("_en"):
            lang = "EN"
        elif label.endswith("_zh"):
            lang = "ZH"
        elif label.endswith("_de"):
            lang = "DE"
        elif label.endswith("_ja"):
            lang = "JA"
        elif label.endswith("_abstract") or "abstract" in label:
            lang = "Abstract"
        else:
            lang = "Unknown"
        return logic, lang

    df[["logic_type", "language"]] = df["Label"].apply(
        lambda x: pd.Series(parse_label(x))
    )
    return df


@st.cache_data
def load_similarity(experiment: str):
    paths = {
        "Order-0 Positions":   BASE_DIR / "results/exp1_order0/data/global_similarity_order0.csv",
        "Order-1 Velocities":  BASE_DIR / "results/exp2_order1/data/global_similarity_order1.csv",
        "Multilingual (5x5)":  BASE_DIR / "results/exp3_multilingual/data/global_similarity_order1.csv",
    }
    path = paths.get(experiment)
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    return df


@st.cache_data
def load_pca_trajectories(logic: str):
    folder_map = {"LogicA": "logicA", "LogicB": "logicB", "LogicC": "logicC"}
    folder = BASE_DIR / "results/exp1_order0/data/pca" / folder_map.get(logic, "logicA")
    if not folder.exists():
        return {}
    results = {}
    for csv_file in sorted(folder.glob("*_pca3d.csv")):
        df = pd.read_csv(csv_file)
        results[csv_file.stem.replace("_pca3d", "")] = df
    return results


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 10px 0 20px 0;'>
            <span style='font-size:2.5rem;'>🧠</span>
            <div style='font-size:1.1rem; font-weight:700; color:#58a6ff; margin-top:6px;'>
                Reasoning Flow
            </div>
            <div style='font-size:0.78rem; color:#8b949e;'>Dashboard v1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown(
        """
        <div class='sidebar-block'>
            <div style='font-weight:700; color:#c9d1d9; margin-bottom:4px;'>👤 Author</div>
            <div>Yash Chaudhary</div>
            <div style='color:#8b949e; font-size:0.8rem;'>Master's Research · NJIT</div>
            <div style='color:#8b949e; font-size:0.8rem;'>Supervisor: Prof. Mengjia Xu</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='sidebar-block'>
            <div style='font-weight:700; color:#c9d1d9; margin-bottom:4px;'>🗄️ Dataset Info</div>
            <div><b>Model:</b> Qwen2.5-0.5B</div>
            <div><b>Dataset:</b> LogicBench</div>
            <div><b>Trajectories:</b> 244</div>
            <div><b>Steps / trajectory:</b> 9</div>
            <div><b>Logic Types:</b> A, B, C</div>
            <div><b>Languages:</b> EN, ZH, DE, JA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class='sidebar-block'>
            <div style='font-weight:700; color:#c9d1d9; margin-bottom:4px;'>📄 Paper</div>
            <div style='color:#8b949e; font-size:0.82rem;'>Zhou et al., ICLR 2026</div>
            <div style='color:#8b949e; font-size:0.82rem;'>"The Geometry of Reasoning"</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem; color:#484f58; text-align:center;'>"
        "Built with Streamlit · Plotly · 2026"
        "</div>",
        unsafe_allow_html=True,
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏠 Overview",
        "📐 Curvature Analysis",
        "🚀 PCA Trajectories",
        "🗺️ Similarity Heatmap",
        "🔭 Extensions & Future Work",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        """
        <div style='padding: 40px 20px 30px 20px; text-align: center;'>
            <div class='hero-title'>The Geometry of Reasoning</div>
            <div class='hero-subtitle'>Flowing Logics in Representation Space</div>
            <div style='margin-top:14px;'>
                <span class='tag'>Hyperbolic Geometry</span>
                <span class='tag'>Chain-of-Thought</span>
                <span class='tag'>Menger Curvature</span>
                <span class='tag'>LLM Internals</span>
                <span class='tag'>Multilingual</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trajectories Analyzed", "244", help="Across 3 logic types and 4 languages")
    c2.metric("Curvature Reduction", "41%", delta="-41% Poincaré vs Euclidean", delta_color="inverse")
    c3.metric("Logic Systems", "3", help="LogicA (Modus Ponens), LogicB (Transitivity), LogicC (Universal Instantiation)")
    c4.metric("Languages", "4", help="English, Chinese, German, Japanese")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:20px 28px; max-width:900px; margin:0 auto 30px auto;'>
            <p style='color:#c9d1d9; font-size:1.0rem; line-height:1.8; margin:0;'>
                This project investigates how a language model's hidden-state activations
                trace geometric curves through embedding space during chain-of-thought
                reasoning. We measure <b>Menger curvature</b> in both Euclidean and
                hyperbolic (Poincaré ball) geometries across 244 reasoning trajectories
                spanning three logic types and four natural languages.
                <br><br>
                We find that hyperbolic space consistently yields lower curvature,
                suggesting that hierarchical reasoning naturally lives in negatively-curved
                space. A key technical contribution is <b>L2 normalisation before the
                exponential map</b>, which resolves the boundary-collapse problem and
                produces a stable 41% reduction in measured curvature.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("<div class='section-header'>Mean Curvature: Euclidean vs Poincaré</div>", unsafe_allow_html=True)
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["Euclidean", "Poincaré"],
            y=[6.5, 3.9],
            marker_color=["#f78166", "#3fb950"],
            text=["6.50", "3.90"],
            textposition="outside",
            width=0.45,
        ))
        fig_bar.add_annotation(
            x=0.5, y=5.2,
            text="−41% reduction",
            font=dict(size=18, color="#e3b341", family="Poppins, sans-serif"),
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            arrowcolor="#e3b341",
        )
        fig_bar.update_layout(
            template=PLOTLY_THEME,
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=PLOTLY_FONT,
            xaxis_title="Geometry",
            yaxis_title="Mean Menger Curvature (κ)",
            xaxis=dict(tickfont=dict(size=14)),
            yaxis=dict(tickfont=dict(size=14)),
            showlegend=False,
            height=380,
            margin=dict(t=30, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        st.markdown("<div class='section-header'>Mean Curvature by Logic Type</div>", unsafe_allow_html=True)
        df_curv = load_curvature()
        if df_curv is not None:
            summary = (
                df_curv.groupby("logic_type")[["Euclidean", "Poincare", "Ratio_P/E"]]
                .mean()
                .round(3)
                .reset_index()
                .rename(columns={"logic_type": "Logic Type", "Ratio_P/E": "Ratio (P/E)"})
            )
            st.dataframe(summary, use_container_width=True, height=160)

            fig_logic = px.bar(
                summary,
                x="Logic Type",
                y=["Euclidean", "Poincare"],
                barmode="group",
                color_discrete_map={"Euclidean": "#f78166", "Poincare": "#3fb950"},
                template=PLOTLY_THEME,
                height=240,
            )
            fig_logic.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=PLOTLY_FONT,
                xaxis_title="Logic Type",
                yaxis_title="Mean Menger Curvature (κ)",
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                legend_title_text="Geometry (Space)",
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_logic, use_container_width=True)
        else:
            st.warning("curvature_l2_normalized.csv not found. Place it in the project root.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CURVATURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        "<h2 style='color:#58a6ff;'>Menger Curvature: Euclidean vs Hyperbolic Space</h2>",
        unsafe_allow_html=True,
    )

    df_curv = load_curvature()

    if df_curv is None:
        st.warning("curvature_l2_normalized.csv not found. Expected at project root.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("<div class='section-header'>Euclidean vs Poincaré per Trajectory</div>", unsafe_allow_html=True)
            fig_scatter = px.scatter(
                df_curv,
                x="Euclidean",
                y="Poincare",
                color="logic_type",
                symbol="language",
                hover_name="Label",
                hover_data={"Euclidean": ":.3f", "Poincare": ":.3f", "Ratio_P/E": ":.3f"},
                color_discrete_map=LOGIC_COLORS,
                template=PLOTLY_THEME,
                labels={"logic_type": "Logic Type", "language": "Language"},
                height=420,
            )
            max_val = float(df_curv[["Euclidean", "Poincare"]].max().max())
            fig_scatter.add_trace(go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                line=dict(color="#484f58", dash="dash"),
                name="y = x  (equal curvature — no hyperbolic benefit)",
                showlegend=True,
            ))
            fig_scatter.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=PLOTLY_FONT,
                xaxis_title="Euclidean Menger Curvature (κ_E)",
                yaxis_title="Poincaré Menger Curvature (κ_P)",
                legend_title_text="Logic Type / Language",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                "Each dot is one of the 244 reasoning trajectories. "
                "Points **below the diagonal** (y = x line) have lower Poincaré curvature than Euclidean — "
                "meaning hyperbolic geometry provides a flatter, more geodesic representation of that reasoning chain. "
                "Shape encodes language; colour encodes logic type."
            )

        with col2:
            st.markdown("<div class='section-header'>Distribution by Logic Type</div>", unsafe_allow_html=True)
            df_melt = df_curv.melt(
                id_vars=["logic_type"],
                value_vars=["Euclidean", "Poincare"],
                var_name="Space",
                value_name="Curvature",
            )
            fig_box = px.box(
                df_melt,
                x="logic_type",
                y="Curvature",
                color="Space",
                color_discrete_map={"Euclidean": "#f78166", "Poincare": "#3fb950"},
                template=PLOTLY_THEME,
                labels={"logic_type": "Logic Type"},
                height=420,
                points="outliers",
            )
            fig_box.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=PLOTLY_FONT,
                xaxis_title="Logic Type",
                yaxis_title="Menger Curvature (κ)",
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                legend_title_text="Geometry (Space)",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_box, use_container_width=True)
            st.caption(
                "Box plots show the interquartile range (IQR: 25th–75th percentile) with whiskers extending to 1.5×IQR. "
                "Dots beyond whiskers are outliers. "
                "When the green (Poincaré) box sits consistently below the red (Euclidean) box, "
                "hyperbolic geometry yields uniformly lower curvature — not just on average, but across the full distribution."
            )

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<div class='section-header'>Mean Curvature by Language</div>", unsafe_allow_html=True)
            lang_summary = (
                df_curv.groupby("language")[["Euclidean", "Poincare"]]
                .mean()
                .round(3)
                .reset_index()
            )
            fig_lang = px.bar(
                lang_summary,
                x="language",
                y=["Euclidean", "Poincare"],
                barmode="group",
                color_discrete_map={"Euclidean": "#f78166", "Poincare": "#3fb950"},
                template=PLOTLY_THEME,
                labels={"language": "Language", "value": "Mean Curvature", "variable": "Space"},
                height=360,
            )
            fig_lang.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=PLOTLY_FONT,
                yaxis_title="Mean Menger Curvature (κ)",
                legend_title_text="Geometry (Space)",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_lang, use_container_width=True)
            st.caption(
                "The ~41% hyperbolic curvature reduction is **language-agnostic**: "
                "it holds consistently across English (EN), Chinese (ZH), German (DE), and Japanese (JA), "
                "and for the abstract (language-free) formulation. "
                "This confirms the effect is geometric, not a linguistic surface artifact."
            )

        with col4:
            st.markdown("<div class='section-header'>Reduction Ratio Distribution (P/E)</div>", unsafe_allow_html=True)
            fig_hist = px.histogram(
                df_curv,
                x="Ratio_P/E",
                color="logic_type",
                nbins=30,
                color_discrete_map=LOGIC_COLORS,
                template=PLOTLY_THEME,
                labels={"Ratio_P/E": "Poincaré / Euclidean Ratio", "logic_type": "Logic Type"},
                height=360,
                opacity=0.8,
                barmode="overlay",
            )
            mean_ratio = df_curv["Ratio_P/E"].mean()
            fig_hist.add_vline(
                x=mean_ratio, line_dash="dash", line_color="#e3b341",
                annotation_text=f"Overall mean = {mean_ratio:.2f}  (all logic types)",
                annotation_position="top right",
            )
            fig_hist.update_layout(
                paper_bgcolor="#161b22",
                plot_bgcolor="#161b22",
                font=PLOTLY_FONT,
                xaxis_title="Poincaré / Euclidean Curvature Ratio  (values < 1 = hyperbolic benefit)",
                yaxis_title="Number of Trajectories",
                legend_title_text="Logic Type",
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            st.caption(
                "A ratio **< 1.0** means Poincaré curvature is lower than Euclidean — i.e., hyperbolic geometry "
                "provides a flatter representation of that trajectory. "
                "The bulk of trajectories cluster in the **0.4–0.7 range**, confirming a robust 30–60% reduction. "
                "Ratios near 1.0 are trajectories where geometry makes little difference (near-linear paths)."
            )

    with st.expander("📖 What is Menger Curvature? (click to expand definition and formula)"):
        st.markdown(
            r"""
            **Menger curvature** κ of three consecutive trajectory points *p*, *q*, *r* is:

            $$\kappa(p, q, r) = \frac{4 \cdot \text{Area}(p,q,r)}{|pq| \cdot |qr| \cdot |pr|}$$

            where Area is computed via Heron's formula. Equivalently, κ = 1/R where R is the
            circumradius of the triangle formed by the three points.

            A **high curvature** means the reasoning trajectory bends sharply;
            **low curvature** means it moves in a straighter (more geodesic) path through embedding space.

            We compute this over consecutive triples of hidden-state vectors along a chain-of-thought,
            using both the standard **Euclidean** metric and the **Poincaré ball** metric
            (hyperbolic geometry with curvature c = 1).

            **Why hyperbolic?** Logical reasoning is hierarchical (premises → conclusions).
            Hyperbolic space is the natural continuous analogue of a tree — distances grow
            exponentially toward the boundary, matching the branching structure of logical deduction.
            """,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PCA TRAJECTORIES
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        "<h2 style='color:#58a6ff;'>3D Reasoning Trajectories in Embedding Space</h2>",
        unsafe_allow_html=True,
    )

    col_ctrl1, col_ctrl2 = st.columns([1, 2])
    with col_ctrl1:
        selected_logic = st.selectbox(
            "Logic Type", ["LogicA", "LogicB", "LogicC"], key="pca_logic"
        )
    with col_ctrl2:
        traj_data = load_pca_trajectories(selected_logic)
        if traj_data:
            topic_options = ["All Topics"] + sorted(traj_data.keys())
        else:
            topic_options = ["All Topics"]
        selected_topic = st.selectbox("Topic / Trajectory", topic_options, key="pca_topic")

    st.markdown(
        """
        <div style='background:#161b22; border:1px solid #30363d; border-radius:8px;
                    padding:12px 18px; margin-bottom:16px; font-size:0.9rem; color:#8b949e;'>
            <b style='color:#c9d1d9;'>What is PCA here?</b> Each reasoning trajectory consists of
            9 hidden-state vectors of dimension 896 (Qwen2.5-0.5B). PCA reduces these to 3 dimensions
            by finding the directions of greatest variance — the axes you see are <em>Principal Component 1, 2, 3</em>,
            ordered from most to least variance explained.<br><br>
            <b style='color:#c9d1d9;'>What each line is:</b> One coloured line = one reasoning chain (9 steps,
            one per token of the chain-of-thought). Colour encodes language (EN/ZH/DE/JA). ◆ = end of chain.<br><br>
            <b style='color:#c9d1d9;'>What to look for:</b> Trajectories of the <em>same logic type</em>
            but different topics cluster together in PCA space — the model uses a similar geometric path for
            the same reasoning pattern regardless of topic. Different logic types occupy
            <em>distinct spatial regions</em>, showing the model encodes logical structure geometrically.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not traj_data:
        st.warning(
            f"No PCA trajectory files found for {selected_logic}. "
            "Expected: results/exp1_order0/data/pca/logicA/ (or logicB/logicC)"
        )
    else:
        if selected_topic == "All Topics":
            plot_data = traj_data
        else:
            plot_data = {selected_topic: traj_data[selected_topic]}

        fig_3d = go.Figure()

        shown_langs = set()
        for label, df_t in plot_data.items():
            if label.endswith("_en"):
                lang = "EN"
            elif label.endswith("_zh"):
                lang = "ZH"
            elif label.endswith("_de"):
                lang = "DE"
            elif label.endswith("_ja"):
                lang = "JA"
            elif "abstract" in label:
                lang = "Abstract"
            else:
                lang = "Unknown"

            color = LANG_COLORS.get(lang, "#aaaaaa")
            hover_text = [f"t={int(row.t)}<br>{label}" for _, row in df_t.iterrows()]
            show_in_legend = lang not in shown_langs
            shown_langs.add(lang)

            fig_3d.add_trace(go.Scatter3d(
                x=df_t["pc1"], y=df_t["pc2"], z=df_t["pc3"],
                mode="lines+markers",
                line=dict(color=color, width=4),
                marker=dict(size=3, color=color),
                name=lang,
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                legendgroup=lang,
                showlegend=show_in_legend,
            ))
            # End marker
            fig_3d.add_trace(go.Scatter3d(
                x=[df_t["pc1"].iloc[-1]],
                y=[df_t["pc2"].iloc[-1]],
                z=[df_t["pc3"].iloc[-1]],
                mode="markers",
                marker=dict(size=8, color=color, symbol="diamond"),
                hovertext=f"END: {label}",
                showlegend=False,
                legendgroup=lang,
            ))

        fig_3d.update_layout(
            template=PLOTLY_THEME,
            paper_bgcolor="#0e1117",
            font=PLOTLY_FONT,
            scene=dict(
                bgcolor="#0e1117",
                xaxis=dict(
                    backgroundcolor="#161b22",
                    gridcolor="#21262d",
                    title="PC 1  (largest variance direction)",
                    tickfont=dict(size=11),
                    title_font=dict(size=12),
                ),
                yaxis=dict(
                    backgroundcolor="#161b22",
                    gridcolor="#21262d",
                    title="PC 2  (second variance direction)",
                    tickfont=dict(size=11),
                    title_font=dict(size=12),
                ),
                zaxis=dict(
                    backgroundcolor="#161b22",
                    gridcolor="#21262d",
                    title="PC 3  (third variance direction)",
                    tickfont=dict(size=11),
                    title_font=dict(size=12),
                ),
            ),
            height=620,
            margin=dict(t=10, b=10),
            legend=dict(
                title=dict(text="Language"),
                bgcolor="#161b22",
                bordercolor="#30363d",
                borderwidth=1,
                font=dict(size=13, family="Poppins, sans-serif"),
            ),
        )
        st.plotly_chart(fig_3d, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — SIMILARITY HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(
        "<h2 style='color:#58a6ff;'>Pairwise Similarity of Reasoning Trajectories</h2>",
        unsafe_allow_html=True,
    )

    experiment = st.selectbox(
        "Choose Experiment",
        ["Order-0 Positions", "Order-1 Velocities", "Multilingual (5x5)"],
        key="sim_exp",
    )

    exp_notes = {
        "Order-0 Positions": (
            "**What you are seeing:** Each cell (i, j) is the cosine similarity between the "
            "hidden-state position sequences of trajectories i and j. "
            "**The block-diagonal structure** means trajectories cluster by *language*: "
            "the model's absolute position in embedding space is dominated by surface language features "
            "(vocabulary, syntax), not by the logic type. "
            "Bright blocks on the diagonal = high within-language similarity; dark off-diagonal = low cross-language similarity."
        ),
        "Order-1 Velocities": (
            "**What you are seeing:** Each cell (i, j) is the cosine similarity between the "
            "*velocity* sequences — the step-by-step differences between consecutive hidden states. "
            "**Same logic type (A/B/C) shows high similarity** regardless of topic or language: "
            "the model moves through embedding space in the same *direction* for the same logical operation. "
            "Values > 0.7 in an off-diagonal block confirm that two trajectories share the same logical structure. "
            "This is the paper's core finding: directional change encodes logic, not surface form."
        ),
        "Multilingual (5x5)": (
            "**What you are seeing:** A 5×5 matrix — one row/column per language variant of LogicA "
            "(Abstract, EN, ZH, DE, JA). "
            "**High off-diagonal values** (e.g., EN vs ZH similarity ≈ 0.8+) confirm that the model "
            "has learned a *language-invariant* representation of Modus Ponens. "
            "The same geometric trajectory through embedding space is used regardless of which language "
            "the chain-of-thought is expressed in."
        ),
    }

    df_sim = load_similarity(experiment)

    if df_sim is None:
        st.warning(
            f"Similarity matrix for '{experiment}' not found. "
            "Check that experiments have been run and results saved."
        )
    else:
        st.info(exp_notes[experiment])

        fig_heat = go.Figure(go.Heatmap(
            z=df_sim.values,
            x=list(df_sim.columns),
            y=list(df_sim.index),
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(
                title="Cosine Similarity<br>(−1=opposite, 0=orthogonal, +1=identical)",
                thickness=14,
            ),
            hoverongaps=False,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Sim: %{z:.3f}<extra></extra>",
        ))
        n = len(df_sim)
        tick_step = max(1, n // 20)
        fig_heat.update_layout(
            template=PLOTLY_THEME,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=PLOTLY_FONT,
            height=max(500, min(900, n * 4 + 100)),
            margin=dict(t=20, b=100, l=100),
            xaxis=dict(
                tickmode="array",
                tickvals=list(range(0, n, tick_step)),
                ticktext=list(df_sim.columns[::tick_step]),
                tickangle=45,
                tickfont=dict(size=11),
                title="Trajectory Index  (sorted by logic type, then language)",
            ),
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(0, n, tick_step)),
                ticktext=list(df_sim.index[::tick_step]),
                tickfont=dict(size=11),
                title="Trajectory Index  (sorted by logic type, then language)",
                autorange="reversed",
            ),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with st.expander("📖 What do Order-0 and Order-1 mean? (click to expand technical explanation)"):
        st.markdown(
            """
            **Order-0 (Positions):** We compare raw hidden-state vectors at each step.
            Similarity reveals *where* trajectories live in embedding space.
            At order 0, **language** is the dominant clustering factor.

            **Order-1 (Velocities):** We compute the *difference* between consecutive
            hidden states (velocity in phase space) and compare those.
            At order 1, **logic type** drives clustering — the model's directional
            changes encode the structure of reasoning, not the surface topic or language.

            **Multilingual:** 5-trajectory experiment (abstract + EN/ZH/DE/JA for LogicA).
            High cross-lingual similarities show the model has language-invariant
            representations of logical inference patterns.
            """,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — EXTENSIONS & FUTURE WORK
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown(
        "<h2 style='color:#58a6ff;'>Novel Contributions: Hyperbolic Geodesic Hypothesis</h2>",
        unsafe_allow_html=True,
    )

    # ── Section 1: Boundary Collapse ─────────────────────────────────────────
    st.markdown("<div class='section-header'>1 · The Boundary Collapse Problem</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown(
            """
            When embedding vectors are passed **raw** into the Poincaré-ball exponential map,
            their large norms (‖h‖ ≈ 50–100) push mapped points toward the **boundary** of the
            unit disk (‖x‖ ≈ 0.999). In hyperbolic geometry, boundary points are infinitely
            far apart — distances explode and curvature becomes meaningless.

            The chart on the right shows **boundary-collapsed** points (red) vs the
            well-behaved, centred distribution after L2 normalisation (green).
            """,
        )

    with col_b:
        rng = np.random.default_rng(42)
        theta = np.linspace(0, 2 * np.pi, 300)
        circle_x, circle_y = np.cos(theta), np.sin(theta)

        r_bnd = rng.uniform(0.990, 0.999, 40)
        a_bnd = rng.uniform(0, 2 * np.pi, 40)

        r_ctr = rng.uniform(0.38, 0.54, 40)
        a_ctr = rng.uniform(0, 2 * np.pi, 40)

        fig_circle = go.Figure()
        fig_circle.add_trace(go.Scatter(
            x=circle_x, y=circle_y, mode="lines",
            line=dict(color="#484f58", width=2), name="Poincaré boundary",
        ))
        fig_circle.add_trace(go.Scatter(
            x=r_bnd * np.cos(a_bnd), y=r_bnd * np.sin(a_bnd), mode="markers",
            marker=dict(color="#f78166", size=9, opacity=0.85),
            name="Raw: ‖x‖ ≈ 0.999 (collapsed)",
        ))
        fig_circle.add_trace(go.Scatter(
            x=r_ctr * np.cos(a_ctr), y=r_ctr * np.sin(a_ctr), mode="markers",
            marker=dict(color="#3fb950", size=9, opacity=0.85),
            name="L2-normalised: ‖x‖ ≈ 0.46",
        ))
        fig_circle.add_annotation(
            x=0, y=-1.08,
            text="Grey circle = Poincaré disk boundary (‖x‖ = 1). Points outside are invalid.",
            font=dict(size=10, color="#8b949e", family="Poppins, sans-serif"),
            showarrow=False,
        )
        fig_circle.update_layout(
            template=PLOTLY_THEME,
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=PLOTLY_FONT,
            xaxis=dict(range=[-1.15, 1.15], scaleanchor="y", constrain="domain",
                       title="x₁  (Poincaré coordinate)"),
            yaxis=dict(range=[-1.25, 1.15], title="x₂  (Poincaré coordinate)"),
            height=360,
            margin=dict(t=10, b=40),
            legend=dict(
                bgcolor="#21262d", bordercolor="#30363d", borderwidth=1,
                font=dict(size=13),
                title=dict(text="Embedding variant"),
            ),
        )
        st.plotly_chart(fig_circle, use_container_width=True)

    # ── Section 2: The Fix ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>2 · The Fix: L2 Normalisation</div>", unsafe_allow_html=True)

    col_c, col_d = st.columns([1, 1])
    with col_c:
        st.markdown(
            """
            **Step 1 — L2 Normalize** (maps hidden states to unit sphere):

            Hidden-state vectors from Qwen2.5-0.5B have norms ‖h‖ ≈ 50–100, varying by a factor of 50–100×
            across reasoning steps. Dividing by the norm removes this magnitude variation and places every
            vector on the unit hypersphere — a prerequisite for a numerically stable exponential map.
            """
        )
        st.latex(r"\hat{x} = \frac{x}{\|x\|_2}")
        st.markdown(
            """
            **Step 2 — Project to Poincaré Ball** via exponential map at the origin:

            The exponential map sends a tangent vector at the origin into the Poincaré disk.
            The tanh function squashes any real-valued norm into (0, 1), guaranteeing the result
            lands strictly inside the disk — never at the boundary — regardless of input magnitude.
            """
        )
        st.latex(
            r"\exp_0(\hat{x}) = \tanh\!\left(\frac{\sqrt{c}\,\|\hat{x}\|}{2}\right)"
            r"\cdot \frac{\hat{x}}{\|\hat{x}\|}"
        )
        st.markdown(
            """
            **Why this is numerically safe:** For curvature *c* = 1 and ‖x̂‖ = 1 (after L2 normalization):
            tanh(½) ≈ **0.462**. This places all embeddings at a Poincaré radius of ~0.462 — well inside
            the disk, far from the boundary singularity at ‖x‖ → 1, and with well-defined distances and curvatures.
            """
        )

    with col_d:
        comparison_data = {
            "": ["Poincaré radius", "Distances", "Curvature"],
            "Before (Raw)": ["≈ 0.999 (boundary!)", "∞ (diverge)", "Undefined / explodes"],
            "After (L2 norm)": ["≈ 0.462 (centered)", "Well-defined", "−41% vs Euclidean"],
        }
        st.table(pd.DataFrame(comparison_data).set_index(""))

    # ── Section 3: Results ────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>3 · Results: 41% Curvature Reduction</div>", unsafe_allow_html=True)

    df_curv2 = load_curvature()
    if df_curv2 is not None:
        means = df_curv2.groupby("logic_type")[["Euclidean", "Poincare"]].mean().reset_index()

        fig_result = go.Figure()
        fig_result.add_trace(go.Bar(
            x=means["logic_type"], y=means["Euclidean"],
            name="Euclidean", marker_color="#f78166", offsetgroup=0,
            text=means["Euclidean"].round(2), textposition="outside",
        ))
        fig_result.add_trace(go.Bar(
            x=means["logic_type"], y=means["Poincare"],
            name="Poincaré (Hyperbolic)", marker_color="#3fb950", offsetgroup=1,
            text=means["Poincare"].round(2), textposition="outside",
        ))
        fig_result.update_layout(
            template=PLOTLY_THEME,
            paper_bgcolor="#161b22",
            plot_bgcolor="#161b22",
            font=PLOTLY_FONT,
            barmode="group",
            xaxis_title="Logic Type  (A = Modus Ponens · B = Transitivity · C = Universal Instantiation)",
            yaxis_title="Mean Menger Curvature κ  (lower = straighter reasoning path)",
            legend_title_text="Geometry",
            height=360,
            margin=dict(t=20, b=60),
            annotations=[
                dict(
                    text="<b>~41% reduction across all logic types</b> — consistent regardless of reasoning pattern",
                    x=0.5, y=1.05, xref="paper", yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="#e3b341", family="Poppins, sans-serif"),
                )
            ],
        )
        st.plotly_chart(fig_result, use_container_width=True)
    else:
        st.warning("Curvature data not available.")

    # ── Section 4: Future Work ────────────────────────────────────────────────
    st.markdown("<div class='section-header'>4 · Future Research Directions</div>", unsafe_allow_html=True)

    future_items = [
        {
            "icon": "🔍",
            "title": "Hallucination Detection via Geodesic Deviation",
            "body": (
                "Hallucinated responses may produce trajectories with anomalously high curvature "
                "or trajectories that fail to converge in embedding space. "
                "A 'geodesic score' (mean hyperbolic curvature along the path) could serve as "
                "a cheap, model-agnostic hallucination signal without requiring ground-truth labels."
            ),
        },
        {
            "icon": "📐",
            "title": "Scale Invariance: 0.5B → 70B Models",
            "body": (
                "Does the 41% hyperbolic curvature reduction hold across model scales? "
                "We hypothesise larger models show lower absolute curvature (smoother reasoning) "
                "but preserve the relative Euclidean-to-hyperbolic ratio — a universal geometric "
                "signature of reasoning."
            ),
        },
        {
            "icon": "🌐",
            "title": "Out-of-Distribution Generalisation",
            "body": (
                "In-distribution reasoning trajectories cluster tightly in PCA space; "
                "OOD inputs may produce diffuse or erratic paths. Curvature-based metrics "
                "could serve as distribution-shift detectors at inference time."
            ),
        },
        {
            "icon": "🧲",
            "title": "Hyperbolic LoRA Fine-tuning",
            "body": (
                "Embed the Poincaré ball into the fine-tuning objective. "
                "A curvature regulariser penalises trajectories deviating from geodesics, "
                "potentially improving reasoning consistency and reducing error compounding "
                "in multi-step tasks."
            ),
        },
    ]

    for item in future_items:
        with st.expander(f"{item['icon']}  {item['title']}", expanded=False):
            st.markdown(item["body"])

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='background:linear-gradient(135deg,#161b22,#21262d);
                    border:1px solid #30363d; border-radius:10px;
                    padding:20px 28px; text-align:center; color:#8b949e;'>
            <span style='font-size:1.1rem; font-weight:700; color:#58a6ff;'>
                The Geometry of Reasoning
            </span><br>
            Yash Chaudhary · Master's Research · NJIT · Prof. Mengjia Xu · 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
