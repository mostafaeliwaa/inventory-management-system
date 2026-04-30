import base64
import json
from datetime import datetime
from io import BytesIO

import pandas as pd
import requests
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Invoice OCR & Product Matching",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ---- Header ---- */
    .inv-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #1d4ed8 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .inv-header::before {
        content: "";
        position: absolute;
        top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,179,237,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .inv-header h1 {
        font-family: 'Syne', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        color: #f0f9ff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .inv-header p {
        color: #93c5fd;
        margin: 0.4rem 0 0;
        font-size: 0.95rem;
        font-weight: 300;
    }

    /* ---- Status dot ---- */
    .dot-green  { color: #22c55e; font-size: 1.1rem; }
    .dot-red    { color: #ef4444; font-size: 1.1rem; }
    .dot-yellow { color: #f59e0b; font-size: 1.1rem; }

    /* ---- Score badges ---- */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-high   { background: #dcfce7; color: #15803d; }
    .badge-medium { background: #fef9c3; color: #a16207; }
    .badge-low    { background: #fee2e2; color: #b91c1c; }

    /* ---- Cards ---- */
    .card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: #94a3b8;
        margin-bottom: 0.2rem;
    }
    .card-value {
        font-size: 1.05rem;
        font-weight: 500;
        color: #0f172a;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: #f1f5f9;
    }

    /* ---- Misc ---- */
    .stButton>button {
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
    }
    hr { border-color: #e2e8f0; }
</style>
""",
    unsafe_allow_html=True,
)
# Session state defaults
def _init_state():
    defaults = {
        "api_base_url": "http://localhost:8000",
        "current_results": None,
        "history": [],       # list[dict]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# API Client


class APIClient:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.timeout = 120

    def health_check(self) -> dict | None:
        try:
            r = requests.get(f"{self.base}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def get_stats(self) -> dict | None:
        try:
            r = requests.get(f"{self.base}/stats", timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def process_invoice(self, file_bytes: bytes, filename: str) -> dict | None:
        try:
            r = requests.post(
                f"{self.base}/process-invoice",
                files={"file": (filename, file_bytes, "application/octet-stream")},
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot reach the API server. Is it running?")
        except requests.exceptions.Timeout:
            st.error("⏱ Request timed out. The invoice might be complex — try again.")
        except requests.exceptions.HTTPError as e:
            detail = ""
            try:
                detail = e.response.json().get("detail", "")
            except Exception:
                pass
            st.error(f"❌ API error {e.response.status_code}: {detail or e}")
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
        return None

    def confirm_items(self, corrections: list[dict]) -> dict | None:
        try:
            r = requests.post(
                f"{self.base}/confirm-items",
                json={"corrections": corrections},
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            st.error(f"❌ Failed to send corrections: {e}")
        return None


@st.cache_resource
def get_client(url: str) -> APIClient:
    return APIClient(url)

# Helper utilities
def _score_badge(score: float | None) -> str:
    if score is None:
        return '<span class="badge badge-low">N/A</span>'
    if score >= 0.8:
        return f'<span class="badge badge-high">{score:.2f}</span>'
    if score >= 0.6:
        return f'<span class="badge badge-medium">{score:.2f}</span>'
    return f'<span class="badge badge-low">{score:.2f}</span>'


def _img_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _add_to_history(filename: str, status: str, result: dict | None):
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "status": status,
        "result": result,
    }
    st.session_state.history.insert(0, entry)
    st.session_state.history = st.session_state.history[:10]


# Sidebar
def render_sidebar(client: APIClient):
    with st.sidebar:
        st.markdown(
            "<p style='font-family:Syne,sans-serif;font-weight:800;font-size:1.1rem;"
            "color:#1e3a5f;margin-bottom:0.2rem;'>⚙️ Control Panel</p>",
            unsafe_allow_html=True,
        )
        st.divider()

        # API status
        health = client.health_check()
        if health:
            dot = "dot-green" if health.get("status") == "ok" else "dot-yellow"
            st.markdown(
                f'<span class="{dot}">●</span> API <b>{"Online" if dot=="dot-green" else "Degraded"}</b>',
                unsafe_allow_html=True,
            )
            g_icon = "✅" if health.get("gemini") else "❌"
            m_icon = "✅" if health.get("matcher") else "❌"
            st.caption(f"Gemini {g_icon}   Matcher {m_icon}")
        else:
            st.markdown('<span class="dot-red">●</span> API **Offline**', unsafe_allow_html=True)

        st.divider()

        # Startup error inspector
        try:
            r = requests.get(f"{client.base}/debug/startup", timeout=5)
            if r.ok:
                dbg = r.json()
                n_errors = len(dbg.get("errors", []))
                if n_errors:
                    with st.expander(f"⚠️ {n_errors} startup error(s)", expanded=False):
                        st.markdown(f"**DB path:** `{dbg.get('product_db_path')}`")
                        st.markdown(f"**DB exists:** `{dbg.get('product_db_exists')}`")
                        for err in dbg.get("errors", []):
                            st.code(err, language="text")
        except Exception:
            pass

        # Stats
        if st.button("🔄 Refresh Stats", width="stretch"):
            st.cache_data.clear()

        stats = client.get_stats()
        if stats:
            # Hide internal matcher statistics from the control panel
            hidden = {
                "total_queries",
                "matches_found",
                "corrections_applied",
                "avg_processing_time",
                "correction_memory_size",
                "database_size",
                "cache_dir",
            }
            public_stats = {k: v for k, v in stats.items() if k not in hidden}
            if public_stats:
                for k, v in public_stats.items():
                    st.metric(k.replace("_", " ").title(), v)
            else:
                st.caption("No public stats available")
        else:
            st.caption("Stats unavailable")

        st.divider()

        # Navigation
        page = st.radio(
            "Navigate",
            ["📄 Dashboard", "🕘 History", "⚙️ Settings"],
            label_visibility="collapsed",
        )
        return page

# Dashboard page
def render_dashboard(client: APIClient):
    st.markdown(
        """
        <div class="inv-header">
            <h1>📄 Invoice OCR &amp; Product Matching</h1>
            <p>Upload an invoice image — we'll extract the data and match every product automatically.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop an invoice image here",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="visible",
    )

    col_img, col_action = st.columns([1, 1], gap="large")

    with col_img:
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, caption=uploaded.name, width="stretch")

    with col_action:
        if uploaded:
            st.markdown("**Ready to process**")
            st.caption(f"`{uploaded.name}` · {uploaded.size / 1024:.1f} KB")

            if st.button("🚀 Process Invoice", type="primary", width="stretch"):
                progress_msgs = [
                    "📤 Uploading image…",
                    "🔍 Running OCR with Gemini…",
                    "🧠 Matching products…",
                    "✅ Finalising results…",
                ]
                bar = st.progress(0)
                status_txt = st.empty()

                import time

                for i, msg in enumerate(progress_msgs):
                    status_txt.markdown(f"*{msg}*")
                    bar.progress((i + 1) * 25)
                    time.sleep(0.4)

                uploaded.seek(0)
                result = client.process_invoice(uploaded.read(), uploaded.name)
                bar.empty()
                status_txt.empty()

                if result:
                    st.session_state.current_results = result
                    _add_to_history(uploaded.name, "success", result)
                    st.success("Processing complete!")
                else:
                    _add_to_history(uploaded.name, "failed", None)

    # ---- Results ----
    results = st.session_state.current_results
    if results:
        st.divider()
        tab_invoice, tab_products, tab_export = st.tabs(
            ["🧾 Invoice Data", "📦 Products", "💾 Export"]
        )

        # ---- Tab 1: Invoice metadata ----
        with tab_invoice:
            inv = results.get("invoice", {})
            if inv:
                cols = st.columns(4)
                fields = [
                    ("Store / Vendor", inv.get("store_name") or inv.get("vendor")),
                    ("Date", inv.get("date") or inv.get("invoice_date")),
                    ("Invoice #", inv.get("invoice_number") or inv.get("invoice_no")),
                    ("Total", inv.get("total") or inv.get("total_amount")),
                ]
                for col, (label, value) in zip(cols, fields):
                    with col:
                        st.markdown(
                            f'<div class="card"><div class="card-label">{label}</div>'
                            f'<div class="card-value">{value or "—"}</div></div>',
                            unsafe_allow_html=True,
                        )
                remaining = {k: v for k, v in inv.items() if k not in
                             {"store_name", "vendor", "date", "invoice_date",
                              "invoice_number", "invoice_no", "total", "total_amount"}}
                if remaining:
                    with st.expander("More invoice fields"):
                        st.json(remaining)
            else:
                st.info("No invoice metadata extracted.")

        # ---- Tab 2: Products ----
        with tab_products:
            products = results.get("products", [])
            if not products:
                st.info("No products found in this invoice.")
            else:
                corrections_to_send: list[dict] = []

                for idx, item in enumerate(products):
                    match = item.get("match") or {}
                    raw_name = item.get("name") or item.get("product_name", "N/A")
                    candidates: list[tuple[str, float]] = []
        
                    item_top = item.get("top_matches") or item.get("top_k") or item.get("candidates")
                    if item_top and isinstance(item_top, (list, tuple)):
                        for x in item_top:
                            if isinstance(x, dict):
                                name = x.get("name") or x.get("matched") or x.get("product")
                                score = x.get("score") or x.get("similarity") or 0.0
                                candidates.append((str(name), float(score)))
                            else:
                                candidates.append((str(x), 0.0))
                    if isinstance(match, dict):
                        for k in ("top_matches", "candidates", "matches", "alternatives", "top_k"):
                            if k in match and isinstance(match[k], (list, tuple)):
                                for x in match[k]:
                                    if isinstance(x, dict):
                                        name = x.get("name") or x.get("matched") or x.get("product")
                                        score = x.get("score") or x.get("similarity") or 0.0
                                        candidates.append((str(name), float(score)))
                                    else:
                                        candidates.append((str(x), 0.0))
                                break
                        if not candidates:
                            for k in ("results", "items"):
                                if k in match and isinstance(match[k], (list, tuple)):
                                    for x in match[k]:
                                        if isinstance(x, dict):
                                            candidates.append((str(x.get("name") or x.get("matched") or ""), float(x.get("score") or 0.0)))
                                        else:
                                            candidates.append((str(x), 0.0))
                                    break

                    best = None
                    if isinstance(match, dict):
                        best = match.get("matched_product") or match.get("best_match") or match.get("best")
                    if not candidates and best:
                        candidates = [str(best)]

                    seen = set()
                    top_candidates: list[tuple[str, float]] = []
                    for name, score in candidates:
                        if name and name not in seen:
                            seen.add(name)
                            top_candidates.append((name, float(score)))
                        if len(top_candidates) >= 10:
                            break
                            
                    with st.container():
                        c1, c2, c3, c4, c5, c6 = st.columns([3, 3, 1.2, 1.2, 1.2, 1.2])
                        c1.markdown(f"**{raw_name}**")

                        selected = None
                        if top_candidates:

                            options = [f"{n} ({s:.2f})" for n, s in top_candidates]
                            radio_options = ["New Item"] + options
                            selected_label = c2.radio("Top matches", options=radio_options, index=0, key=f"sel_{idx}")
                            if selected_label == "New Item":
                                entered = c2.text_input("Enter new product", value="", key=f"other_{idx}")
                                selected = entered.strip() if entered else ""
                            elif selected_label and selected_label != "(none)":
                                selected = selected_label.rsplit(" (", 1)[0]
                        else:
                            selected = c2.text_input("Match", value=str(best or ""), key=f"sel_{idx}")

                        score = match.get("score") or match.get("similarity") if isinstance(match, dict) else None
                        c3.markdown(_score_badge(score), unsafe_allow_html=True)

                        qty_val = item.get("quantity") or item.get("qty") or ""
                        unit_val = item.get("unit_price") or item.get("price") or ""
                        q = c4.text_input("Qty", value=str(qty_val), key=f"qty_{idx}")
                        u = c5.text_input("Unit", value=str(unit_val), key=f"unit_{idx}")
                        try:
                            total = float(q) * float(u)
                            total_str = f"{total:.2f}"
                        except Exception:
                            total_str = ""
                        c6.markdown(f"**Total:** {total_str}")

                        action_col = st.columns([1])[0]
                        if action_col.button("Save / Confirm", key=f"save_{idx}"):

                            correct_text = selected if isinstance(selected, str) and selected != "(none)" else (str(best) if best else "")
                            if correct_text:
                                corrections_to_send.append({"wrong_text": raw_name, "correct_text": correct_text})
                            if "edited_products" not in st.session_state:
                                st.session_state.edited_products = {}
                            st.session_state.edited_products[idx] = {"quantity": q, "unit_price": u, "total": total_str}

                    st.divider()

                if corrections_to_send:
                    resp = client.confirm_items(corrections_to_send)
                    if resp:
                        st.success(f"✅ {resp.get('applied', 0)} correction(s) saved.")

        # ---- Tab 3: Export ----
        with tab_export:
            json_str = json.dumps(results, ensure_ascii=False, indent=2)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "⬇️ Download JSON",
                data=json_str,
                file_name=f"invoice_{ts}.json",
                mime="application/json",
                width="stretch",
            )

            products = results.get("products", [])
            if products:
                rows = []
                for item in products:
                    match = item.get("match") or {}
                    rows.append({
                        "Extracted Name": item.get("name") or item.get("product_name", ""),
                        "Matched Product": match.get("matched_product") or match.get("best_match", ""),
                        "Score": match.get("score") or match.get("similarity", ""),
                        "Quantity": item.get("quantity") or item.get("qty", ""),
                        "Unit Price": item.get("unit_price") or item.get("price", ""),
                    })
                df = pd.DataFrame(rows)
                st.download_button(
                    "⬇️ Download Products CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name=f"products_{ts}.csv",
                    mime="text/csv",
                    width="stretch",
                )

# History page
def render_history():
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;font-weight:700;'>🕘 Processing History</h2>",
        unsafe_allow_html=True,
    )

    history = st.session_state.history
    if not history:
        st.info("No invoices processed yet in this session.")
        return

    for i, entry in enumerate(history):
        status_icon = "✅" if entry["status"] == "success" else "❌"
        with st.expander(f"{status_icon}  {entry['filename']}  —  {entry['timestamp']}"):
            if entry["result"]:
                inv = entry["result"].get("invoice", {})
                if inv:
                    st.json(inv)
                products = entry["result"].get("products", [])
                st.caption(f"{len(products)} product(s) extracted.")
                if st.button("Load this result", key=f"hist_{i}"):
                    st.session_state.current_results = entry["result"]
                    st.success("Result loaded — switch to Dashboard to view.")
            else:
                st.warning("Processing failed for this file.")

# Settings page
def render_settings():
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;font-weight:700;'>⚙️ Settings</h2>",
        unsafe_allow_html=True,
    )

    st.subheader("API Configuration")
    new_url = st.text_input("API Base URL", value=st.session_state.api_base_url)
    if st.button("Save URL"):
        st.session_state.api_base_url = new_url.rstrip("/")
        st.cache_resource.clear()
        st.success("URL updated.")

    st.divider()
    st.subheader("Match Threshold (UI reference only)")
    st.slider(
        "Minimum match score to show a result",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="This slider is informational. Actual threshold is set in the backend configuration.",
    )
    st.caption("To change the backend threshold, update `ProductMatcher` initialization in `api/main.py`.")

# Main entrypoint
def main():
    client = get_client(st.session_state.api_base_url)
    page = render_sidebar(client)

    if "Dashboard" in page:
        render_dashboard(client)
    elif "History" in page:
        render_history()
    elif "Settings" in page:
        render_settings()


if __name__ == "__main__":
    main()
