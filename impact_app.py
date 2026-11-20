import os
import re
import warnings
from datetime import date
 
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
 
# ---------- Setup ----------
load_dotenv()
MODEL = "gemini-3.5-flash"
MAX_SUMMARY_CHARS = 12000
 
AUDIENCES = {
    "Donors 💝": "Warm and story-driven. Celebrate the impact, thank supporters, "
                 "and end with a gentle call to keep supporting the work.",
    "Board / leadership 📋": "Concise, professional and data-forward. Highlight trends, "
                            "risks and any decisions or resources needed.",
    "Volunteers newsletter 🙌": "Friendly and celebratory. Thank volunteers by role "
                               "(not by name) and show what their time made possible.",
    "Grant application 📝": "Formal and outcomes-focused. Emphasise measurable results, "
                           "reach and evidence of effectiveness.",
    "Social media post 📱": "SOCIAL",
}
 
 
# ---------- Helpers: data ----------
def get_api_key():
    # On your Mac: read from .env. Online: read from Streamlit Secrets.
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        try:
            key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            key = None
    return key
 
 
def load_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)
 
 
def make_sample_data():
    rng = np.random.default_rng(7)
    rows = []
    for week in pd.date_range("2026-01-04", "2026-06-28", freq="W"):
        for centre in ["Hyderabad", "Vijayawada", "Chennai", "Pune"]:
            rows.append({
                "date": week.strftime("%Y-%m-%d"),
                "centre": centre,
                "program": rng.choice(["Literacy", "Math", "Life Skills"]),
                "sessions_held": int(rng.integers(1, 4)),
                "children_attended": int(rng.integers(15, 45)),
                "volunteers": int(rng.integers(2, 8)),
            })
    return pd.DataFrame(rows)
 
 
def clean_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all")  # drop fully empty rows
    df = df.loc[:, df.notna().any()]  # drop fully empty columns
    return df
 
 
def find_date_columns(df):
    """Find columns that look like dates and convert them in place."""
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
                date_cols.append(col)
    return date_cols
 
 
def numeric_columns(df):
    return list(df.select_dtypes(include="number").columns)
 
 
def category_columns(df, date_cols):
    """Text columns with a manageable number of repeated values (e.g. centre, program)."""
    nums = numeric_columns(df)
    result = []
    for col in df.columns:
        if col in nums or col in date_cols:
            continue
        unique = df[col].nunique()
        if 1 < unique <= 30 and unique < len(df):
            result.append(col)
    return result
 
 
def fmt(x):
    if pd.isna(x):
        return "n/a"
    if float(x).is_integer():
        return f"{int(x):,}"
    return f"{x:,.2f}"
 
 
def build_summary(df, date_cols):
    """Turn the data into plain-text statistics. Only this is sent to the AI."""
    nums = numeric_columns(df)[:10]
    cats = category_columns(df, date_cols)
    lines = [f"Number of records (rows): {len(df):,}", f"Columns: {', '.join(df.columns)}", ""]
 
    if nums:
        lines.append("NUMERIC COLUMNS:")
        for col in nums:
            s = df[col].dropna()
            if not s.empty:
                lines.append(f"- {col}: total {fmt(s.sum())}, average {fmt(s.mean())}, "
                             f"min {fmt(s.min())}, max {fmt(s.max())}")
        lines.append("")
 
    for col in date_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        lines.append(f"DATE COLUMN '{col}': from {s.min().date()} to {s.max().date()}")
        monthly = df.dropna(subset=[col]).set_index(col).resample("MS")
        table = monthly[nums].sum() if nums else monthly.size().to_frame("records")
        if nums:
            table.insert(0, "records", monthly.size())
        table.index = table.index.strftime("%Y-%m")
        lines.append(f"Monthly totals by '{col}':\n{table.tail(24).to_string()}")
        lines.append("")
 
    for col in cats:
        counts = df[col].value_counts().head(10)
        lines.append(f"CATEGORY '{col}' (records per value): "
                     + ", ".join(f"{k} ({v})" for k, v in counts.items()))
        if nums:
            grouped = df.groupby(col)[nums].sum().head(15)
            lines.append(f"Totals by '{col}':\n{grouped.to_string()}")
        lines.append("")
 
    skipped = [c for c in df.columns if c not in nums and c not in date_cols and c not in cats]
    if skipped:
        lines.append("Not summarised (free text or unique values, e.g. names or notes): "
                     + ", ".join(skipped))
 
    summary = "\n".join(lines)
    if len(summary) > MAX_SUMMARY_CHARS:
        summary = summary[:MAX_SUMMARY_CHARS] + "\n...(summary truncated)"
    return summary
 
 
def monthly_totals(df, date_col, value_col=None):
    monthly = df.dropna(subset=[date_col]).set_index(date_col).resample("MS")
    series = monthly.size() if value_col is None else monthly[value_col].sum()
    series.index = series.index.strftime("%Y-%m")
    return series
 
 
def quick_insights(df, date_cols, main_measure, cats):
    """Simple facts the app calculates itself (no AI needed)."""
    insights = []
    label = main_measure or "records"
    if date_cols:
        series = monthly_totals(df, date_cols[0], main_measure)
        if len(series) >= 2:
            insights.append(f"📅 Best month for **{label}**: **{series.idxmax()}** ({fmt(series.max())})")
            insights.append(f"📉 Quietest month: **{series.idxmin()}** ({fmt(series.min())})")
            change = series.iloc[-1] - series.iloc[-2]
            if series.iloc[-2]:
                pct = change / series.iloc[-2] * 100
                arrow = "📈 up" if change >= 0 else "📉 down"
                insights.append(f"{arrow} **{abs(pct):.0f}%** in {series.index[-1]} vs {series.index[-2]}")
    for col in cats[:2]:
        groups = df.groupby(col)[main_measure].sum() if main_measure else df[col].value_counts()
        if len(groups) >= 2 and groups.max() != groups.min():
            insights.append(f"🏆 Top **{col}**: **{groups.idxmax()}** ({fmt(groups.max())} {label})")
            insights.append(f"🌱 Room to grow: **{groups.idxmin()}** ({fmt(groups.min())} {label})")
    return insights
 
 
def bold_to_html(text):
    return re.sub(r"[*][*](.+?)[*][*]", r"<b>\1</b>", text)
 
 
# ---------- Helpers: AI ----------
def get_client():
    api_key = get_api_key()
    if not api_key:
        raise ValueError("No API key found. Add GOOGLE_API_KEY to .env (on your Mac) or to Secrets (online).")
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=90000))
 
 
def write_report(summary, sample_rows, org, program, period, audience, length, context):
    client = get_client()
 
    if AUDIENCES[audience] == "SOCIAL":
        format_rules = (
            "Write TWO pieces in Markdown:\n"
            "1. '## LinkedIn post': a professional but warm post (max 150 words) with 2-3 key numbers.\n"
            "2. '## Instagram caption': a short, upbeat caption with emojis and 5-8 relevant hashtags.\n"
        )
    else:
        detail = "about 250 words" if length == "Short" else "about 600 words"
        format_rules = (
            f"Tone and audience: {AUDIENCES[audience]}\n"
            f"Length: {detail}.\n"
            "Format in Markdown with these sections: a title, a one-paragraph headline summary, "
            "'Key highlights' (bullets with exact numbers), 'What the data shows' (trends, "
            "top and lowest performers, comparisons), 'Challenges or gaps' (only what the data "
            "actually suggests), 'Next steps', and a closing line suited to the audience.\n"
        )
 
    prompt = (
        "You are an experienced nonprofit communications and impact-reporting writer.\n"
        "Write based ONLY on the data summary and context below.\n"
        "STRICT RULES: never invent numbers, names, stories, quotes or facts that are not in the "
        "summary or context. Use the exact figures given. If something is unclear, describe it "
        "generally instead of guessing. Do not mention individuals.\n\n"
        f"Organization: {org or 'not specified'}\n"
        f"Program: {program or 'not specified'}\n"
        f"Reporting period: {period or 'use the date range in the data, if any'}\n"
        f"Extra context from the user: {context or 'none'}\n\n"
        f"{format_rules}\n"
        f"DATA SUMMARY:\n{summary}\n"
    )
    if sample_rows:
        prompt += f"\nSAMPLE ROWS (for context only):\n{sample_rows}\n"
 
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text
 
 
def revise_report(report, instruction):
    client = get_client()
    prompt = (
        f"Revise the impact report below. Instruction: {instruction}\n"
        "Keep every number exactly as it is. Do not add new facts, numbers, names or stories. "
        "Return only the revised report in Markdown.\n\n"
        f"REPORT:\n{report}"
    )
    response = client.models.generate_content(model=MODEL, contents=prompt)
    return response.text
 
 
# ---------- Look and feel ----------
st.set_page_config(page_title="Impact Report Writer", page_icon="🌱", layout="wide")
 
st.markdown(
    """
    <style>
    .hero {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 55%, #f59e0b 100%);
        padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(15, 118, 110, 0.25);
    }
    .hero h1 { color: white !important; margin: 0; font-size: 2.4rem; }
    .hero p { color: #f0fdfa; margin: 0.4rem 0 0; font-size: 1.1rem; }
    div[data-testid="stMetric"] {
        background: rgba(20, 184, 166, 0.08); border: 1px solid rgba(20, 184, 166, 0.35);
        border-radius: 14px; padding: 0.9rem 1rem;
    }
    .stButton > button[kind="primary"], .stDownloadButton > button {
        background: linear-gradient(90deg, #0d9488, #f59e0b); color: white; border: none;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
        filter: brightness(1.08); color: white;
    }
    .insight {
        background: rgba(245, 158, 11, 0.10); border-left: 5px solid #f59e0b;
        padding: 0.7rem 1rem; border-radius: 8px; margin: 0.4rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
 
st.markdown(
    """
    <div class="hero">
        <h1>🌱 Impact Report Writer</h1>
        <p>Turn your program data into charts, insights and a ready-to-share impact story.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "🔒 **Privacy:** please remove names, phone numbers and other personal details before "
    "uploading. Only summary statistics are sent to the AI, unless you choose to include sample rows."
)
 
# ---------- 1. Get the data ----------
st.header("1️⃣ Your data")
uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])
if st.button("No data handy? Try sample data"):
    st.session_state.use_sample = True
 
df = None
if uploaded is not None:
    try:
        df = load_file(uploaded)
        st.session_state.use_sample = False
    except Exception as error:
        st.error(f"Couldn't read that file. Is it a valid CSV or Excel file? ({error})")
elif st.session_state.get("use_sample"):
    df = make_sample_data()
    st.caption("Using sample data from an imaginary education NGO.")
 
if df is None:
    st.stop()
 
df = clean_columns(df)
if df.empty:
    st.warning("That file looks empty. Please check it and try again.")
    st.stop()
 
date_cols = find_date_columns(df)
nums = numeric_columns(df)
cats = category_columns(df, date_cols)
 
# ---------- Sidebar: interactive filters ----------
with st.sidebar:
    st.header("🎛️ Explore your data")
    people_words = ("child", "attend", "student", "benefic", "particip", "people", "reach")
    preferred = next((i for i, c in enumerate(nums) if any(w in c.lower() for w in people_words)), 0)
    main_measure = st.selectbox(
        "Main measure", nums if nums else ["(record count)"], index=preferred,
        help="The number the insights and charts focus on, e.g. children attended.",
    )
    main_measure = None if main_measure == "(record count)" else main_measure
 
    filtered = df.copy()
    filter_notes = []
 
    if date_cols:
        dc = date_cols[0]
        start, end = df[dc].min().date(), df[dc].max().date()
        if start < end:
            picked = st.slider("Date range", min_value=start, max_value=end, value=(start, end))
            filtered = filtered[(filtered[dc].dt.date >= picked[0]) & (filtered[dc].dt.date <= picked[1])]
            if picked != (start, end):
                filter_notes.append(f"{dc} from {picked[0]} to {picked[1]}")
 
    for col in cats[:3]:
        options = sorted(df[col].dropna().astype(str).unique())
        chosen = st.multiselect(f"Filter by {col}", options, default=options)
        filtered = filtered[filtered[col].astype(str).isin(chosen)]
        if len(chosen) < len(options):
            filter_notes.append(f"{col} = {', '.join(chosen)}")
 
    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} records")
    st.divider()
    st.caption("Built with 💚 by Hema using Python, pandas, Streamlit and Gemini.")
 
if filtered.empty:
    st.warning("No records match these filters. Widen them in the sidebar. 👈")
    st.stop()
 
# ---------- 2. Quick look ----------
st.header("2️⃣ Quick look")
metric_cols = st.columns(min(len(nums), 4) + 1)
metric_cols[0].metric("Records", f"{len(filtered):,}")
for i, col in enumerate(nums[:4], start=1):
    delta = None
    if date_cols:
        series = monthly_totals(filtered, date_cols[0], col)
        if len(series) >= 2:
            change = series.iloc[-1] - series.iloc[-2]
            delta = f"{'+' if change >= 0 else ''}{fmt(change)} last month"
    metric_cols[i].metric(f"Total {col}", fmt(filtered[col].sum()), delta=delta)
 
insights = quick_insights(filtered, date_cols, main_measure, cats)
if insights:
    st.subheader("💡 Quick insights")
    ins_left, ins_right = st.columns(2)
    for i, text in enumerate(insights):
        (ins_left if i % 2 == 0 else ins_right).markdown(
            f'<div class="insight">{bold_to_html(text)}</div>',
            unsafe_allow_html=True,
        )
 
tab_names = []
if date_cols:
    tab_names.append("📈 Trends")
if cats:
    tab_names.append("📊 Compare groups")
tab_names.append("🔍 Data table")
tabs = dict(zip(tab_names, st.tabs(tab_names)))
measure_options = ["(record count)"] + nums
default_index = measure_options.index(main_measure) if main_measure else 0
 
if "📈 Trends" in tabs:
    with tabs["📈 Trends"]:
        c1, c2, c3 = st.columns(3)
        date_col = c1.selectbox("Date column", date_cols)
        value_col = c2.selectbox("Value", measure_options, index=default_index, key="trend_value")
        style = c3.radio("Chart style", ["Line", "Area", "Bars"], horizontal=True)
        series = monthly_totals(filtered, date_col, None if value_col == "(record count)" else value_col)
        if style == "Line":
            st.line_chart(series, color="#0d9488")
        elif style == "Area":
            st.area_chart(series, color="#14b8a6")
        else:
            st.bar_chart(series, color="#0d9488")
 
if "📊 Compare groups" in tabs:
    with tabs["📊 Compare groups"]:
        c1, c2, c3 = st.columns(3)
        cat_col = c1.selectbox("Group by", cats)
        bar_value = c2.selectbox("Value", measure_options, index=default_index, key="bar_value")
        top_n = c3.slider("Show top", 3, 15, 10)
        if bar_value == "(record count)":
            bars = filtered[cat_col].value_counts()
        else:
            bars = filtered.groupby(cat_col)[bar_value].sum().sort_values(ascending=False)
        st.bar_chart(bars.head(top_n), color="#f59e0b", horizontal=True)
 
with tabs["🔍 Data table"]:
    search = st.text_input("🔎 Search the table", placeholder="Type to filter rows...")
    table = filtered
    if search:
        mask = filtered.astype(str).apply(lambda row: row.str.contains(search, case=False, regex=False).any(), axis=1)
        table = filtered[mask]
    st.dataframe(table, use_container_width=True, height=350)
    st.caption(f"{len(table):,} rows")
 
# ---------- 3. Write the report ----------
st.header("3️⃣ Write the report")
form_left, form_right = st.columns(2)
with form_left:
    org = st.text_input("Organization name", placeholder="e.g. Make a Difference")
    program = st.text_input("Program name", placeholder="e.g. After-school literacy program")
    period = st.text_input("Reporting period", placeholder="e.g. January to June 2026")
with form_right:
    audience = st.selectbox("Who is this report for?", list(AUDIENCES.keys()))
    length = st.radio("Length", ["Short", "Detailed"], horizontal=True)
    include_rows = st.checkbox("Also send 10 sample rows to the AI (only if they contain no personal data)")
context = st.text_area(
    "Anything else the AI should know? (optional)",
    placeholder="e.g. We opened a new centre in March. Exams happened in April.",
)
if filter_notes:
    st.caption("🎛️ The report will only use the filtered data: " + "; ".join(filter_notes))
 
if st.button("✨ Write my impact report", type="primary", use_container_width=True):
    summary = build_summary(filtered, date_cols)
    if filter_notes:
        context = f"{context}\nNote: data is filtered to {'; '.join(filter_notes)}."
    sample = filtered.head(10).to_string() if include_rows else ""
    with st.spinner("Crunching numbers and finding the story in your data..."):
        try:
            st.session_state.report = write_report(
                summary, sample, org, program, period, audience, length, context
            )
            st.session_state.summary = summary
            st.toast("Your report is ready! 🎉")
        except Exception as error:
            st.error(f"Something went wrong while writing the report: {error}")
 
if st.session_state.get("report"):
    st.divider()
    st.subheader("📝 Your report")
 
    st.write("Not quite right? Tweak it with one click:")
    tweaks = {
        "✂️ Shorter": "Make it about half as long, keeping the most important points.",
        "💖 More heartfelt": "Make it warmer and more emotional, focusing on the human impact.",
        "📊 More data-focused": "Make it more analytical, leading with numbers and trends.",
        "🎉 More upbeat": "Make the tone more celebratory and energetic.",
    }
    tweak_cols = st.columns(len(tweaks))
    for col, (label, instruction) in zip(tweak_cols, tweaks.items()):
        if col.button(label, use_container_width=True):
            with st.spinner("Rewriting..."):
                try:
                    st.session_state.report = revise_report(st.session_state.report, instruction)
                    st.toast("Updated! ✨")
                except Exception as error:
                    st.error(f"Couldn't rewrite the report: {error}")
 
    with st.container(border=True):
        st.markdown(st.session_state.report)
 
    dl_left, dl_right = st.columns(2)
    dl_left.download_button(
        "⬇️ Download report (.md)",
        st.session_state.report,
        file_name=f"impact_report_{date.today()}.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dl_right.download_button(
        "⬇️ Download report (.txt)",
        st.session_state.report,
        file_name=f"impact_report_{date.today()}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    with st.expander("See the exact numbers the AI was given"):
        st.code(st.session_state.get("summary", ""), language=None)
    st.caption("AI can make mistakes. Check key figures against your data before sharing.")
 