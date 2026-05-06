import streamlit as st
from backend import generate_strategy

st.set_page_config(
    page_title="AI Sports Strategy Generator",
    page_icon="🏀",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef4ff 100%);
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 920px;
    }

    .title {
        font-size: 3.1rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-size: 1.08rem;
        color: #475569;
        margin-bottom: 1.8rem;
    }

    .card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid #dbe7ff;
        box-shadow: 0 12px 30px rgba(37, 99, 235, 0.08);
        margin-bottom: 1.25rem;
    }

    .section-title {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .small-label {
        color: #334155;
        font-size: 0.95rem;
        margin-bottom: 0.45rem;
        font-weight: 600;
    }

    .report-box {
        background: #ffffff;
        padding: 1.4rem;
        border-radius: 18px;
        border: 1px solid #dbe7ff;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.07);
    }

    .report-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 1rem;
    }

    .report-section {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
    }

    .report-label {
        font-size: 0.84rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #2563eb;
        margin-bottom: 0.35rem;
    }

    .report-text {
        color: #1e293b;
        font-size: 1rem;
        line-height: 1.75;
    }

    .matchup-text {
        color: #0f172a;
        font-size: 1.02rem;
        font-weight: 600;
        line-height: 1.6;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 14px;
        height: 3.2rem;
        font-size: 1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #2563eb, #3b82f6);
        color: white;
        border: none;
        box-shadow: 0 10px 18px rgba(37, 99, 235, 0.18);
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8, #2563eb);
        color: white;
    }

    div[data-baseweb="select"] > div {
        color:#0f172a !important;
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


def parse_strategy_output(strategy_text: str):
    lines = [line.strip() for line in strategy_text.splitlines() if line.strip()]

    matchup = ""
    defense = ""
    defense_why = ""
    offense = ""
    offense_why = ""
    rebound_margin = ""
    ast_tov = ""

    for line in lines:
        lower = line.lower()
        if "matchup strategy analysis" in lower:
            matchup = line.replace("🏀", "").strip()
        elif lower.startswith("rebounding margin:"):
            rebound_margin = line.split(":", 1)[1].strip()
        elif lower.startswith("assist-to-turnover ratio:"):
            ast_tov = line.split(":", 1)[1].strip()
        elif lower.startswith("defensive recommendation:"):
            defense = line.split(":", 1)[1].strip()
        elif lower.startswith("why:") and defense and not defense_why:
            defense_why = line.split(":", 1)[1].strip()
        elif lower.startswith("offensive recommendation:"):
            offense = line.split(":", 1)[1].strip()
        elif lower.startswith("why:") and offense and defense_why:
            offense_why = line.split(":", 1)[1].strip()

    return {
        "matchup": matchup,
        "rebound_margin": rebound_margin,
        "ast_tov": ast_tov,
        "defense": defense,
        "defense_why": defense_why,
        "offense": offense,
        "offense_why": offense_why,
    }


# ---------- Header ----------
st.markdown('<div class="title">🏀 AI Sports Strategy Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Select two NBA teams and generate offensive and defensive strategy recommendations using historical NBA data.</div>',
    unsafe_allow_html=True
)

nba_teams = [
    "Atlanta Hawks",
    "Boston Celtics",
    "Brooklyn Nets",
    "Charlotte Hornets",
    "Chicago Bulls",
    "Cleveland Cavaliers",
    "Dallas Mavericks",
    "Denver Nuggets",
    "Detroit Pistons",
    "Golden State Warriors",
    "Houston Rockets",
    "Indiana Pacers",
    "LA Clippers",
    "Los Angeles Lakers",
    "Memphis Grizzlies",
    "Miami Heat",
    "Milwaukee Bucks",
    "Minnesota Timberwolves",
    "New Orleans Pelicans",
    "New York Knicks",
    "Oklahoma City Thunder",
    "Orlando Magic",
    "Philadelphia 76ers",
    "Phoenix Suns",
    "Portland Trail Blazers",
    "Sacramento Kings",
    "San Antonio Spurs",
    "Toronto Raptors",
    "Utah Jazz",
    "Washington Wizards",
]

# ---------- Input Card ----------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Matchup Selection</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="small-label">Home Team</div>', unsafe_allow_html=True)
    team_1 = st.selectbox("Team 1", nba_teams, index=0, label_visibility="collapsed")

with col2:
    st.markdown('<div class="small-label">Away Team</div>', unsafe_allow_html=True)
    team_2 = st.selectbox("Team 2", nba_teams, index=1, label_visibility="collapsed")

generate = st.button("Generate Strategy")

st.markdown('</div>', unsafe_allow_html=True)

# ---------- Result Card ----------
if generate:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Generated Strategy</div>', unsafe_allow_html=True)

    if team_1 == team_2:
        st.error("Please select two different teams.")
    else:
        with st.spinner("Analysing matchup and generating strategy..."):
            strategy = generate_strategy(team_1, team_2)

        parsed = parse_strategy_output(strategy)

        st.markdown('<div class="report-box">', unsafe_allow_html=True)
        st.markdown(
            f'<div class="report-title">Match Report</div>'
            f'<div class="matchup-text">{parsed["matchup"]}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="report-section">'
            f'<div class="report-label">Key Metrics</div>'
            f'<div class="report-text"><strong>Rebounding Margin:</strong> {parsed["rebound_margin"]}<br>'
            f'<strong>Assist-to-Turnover Ratio:</strong> {parsed["ast_tov"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="report-section">'
            f'<div class="report-label">Defensive Recommendation</div>'
            f'<div class="report-text"><strong>{parsed["defense"]}</strong><br>{parsed["defense_why"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="report-section">'
            f'<div class="report-label">Offensive Recommendation</div>'
            f'<div class="report-text"><strong>{parsed["offense"]}</strong><br>{parsed["offense_why"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
