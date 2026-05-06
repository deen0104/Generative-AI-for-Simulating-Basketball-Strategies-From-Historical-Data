import pandas as pd
from functools import lru_cache
from io import StringIO
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

TEAM_NAME_MAP = {
    "Atlanta Hawks": "Hawks",
    "Boston Celtics": "Celtics",
    "Brooklyn Nets": "Nets",
    "Charlotte Hornets": "Hornets",
    "Chicago Bulls": "Bulls",
    "Cleveland Cavaliers": "Cavaliers",
    "Dallas Mavericks": "Mavericks",
    "Denver Nuggets": "Nuggets",
    "Detroit Pistons": "Pistons",
    "Golden State Warriors": "Warriors",
    "Houston Rockets": "Rockets",
    "Indiana Pacers": "Pacers",
    "LA Clippers": "Clippers",
    "Los Angeles Lakers": "Lakers",
    "Memphis Grizzlies": "Grizzlies",
    "Miami Heat": "Heat",
    "Milwaukee Bucks": "Bucks",
    "Minnesota Timberwolves": "Timberwolves",
    "New Orleans Pelicans": "Pelicans",
    "New York Knicks": "Knicks",
    "Oklahoma City Thunder": "Thunder",
    "Orlando Magic": "Magic",
    "Philadelphia 76ers": "76ers",
    "Phoenix Suns": "Suns",
    "Portland Trail Blazers": "Trail Blazers",
    "Sacramento Kings": "Kings",
    "San Antonio Spurs": "Spurs",
    "Toronto Raptors": "Raptors",
    "Utah Jazz": "Jazz",
    "Washington Wizards": "Wizards",
}


def first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


@lru_cache(maxsize=1)
def load_and_prepare_data():
    games = pd.read_csv("games.csv")
    details = pd.read_csv("games_details.csv")
    teams = pd.read_csv("teams.csv")

    games.columns = games.columns.str.upper().str.strip()
    details.columns = details.columns.str.upper().str.strip()
    teams.columns = teams.columns.str.upper().str.strip()

    rename_map = {}
    if "VISITOR_TEAM_ID" not in games.columns and "AWAY_TEAM_ID" in games.columns:
        rename_map["AWAY_TEAM_ID"] = "VISITOR_TEAM_ID"
    if "GAME_ID" not in games.columns and "GAMEID" in games.columns:
        rename_map["GAMEID"] = "GAME_ID"
    games = games.rename(columns=rename_map)

    pts_home_col = first_existing(games.columns, ["PTS_HOME", "HOME_PTS", "PTSHOME"])
    pts_away_col = first_existing(games.columns, ["PTS_AWAY", "AWAY_PTS", "PTSAWAY"])

    if pts_home_col and pts_home_col != "PTS_HOME":
        games = games.rename(columns={pts_home_col: "PTS_HOME"})
    if pts_away_col and pts_away_col != "PTS_AWAY":
        games = games.rename(columns={pts_away_col: "PTS_AWAY"})

    possible_stats = ["FG_PCT", "FG3_PCT", "FT_PCT", "REB", "AST", "TOV", "PTS"]
    available_stats = [c for c in possible_stats if c in details.columns]

    team_avg = (
        details[["GAME_ID", "TEAM_ID"] + available_stats]
        .groupby(["GAME_ID", "TEAM_ID"], as_index=False)[available_stats]
        .mean(numeric_only=True)
    )

    merged = games.merge(
        team_avg,
        how="left",
        left_on=["GAME_ID", "HOME_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
    )

    for c in available_stats:
        if c in merged.columns:
            merged = merged.rename(columns={c: f"{c}_HOME"})

    merged = merged.drop(columns=["TEAM_ID"], errors="ignore")

    merged = merged.merge(
        team_avg,
        how="left",
        left_on=["GAME_ID", "VISITOR_TEAM_ID"],
        right_on=["GAME_ID", "TEAM_ID"],
        suffixes=("", "_AWAY"),
    )

    for c in available_stats:
        if f"{c}_AWAY" not in merged.columns and c in merged.columns:
            merged = merged.rename(columns={c: f"{c}_AWAY"})

    merged = merged.drop(columns=["TEAM_ID"], errors="ignore")

    merged = merged.merge(
        teams[["TEAM_ID", "NICKNAME"]],
        how="left",
        left_on="HOME_TEAM_ID",
        right_on="TEAM_ID",
    ).rename(columns={"NICKNAME": "HOME_TEAM_NAME"}).drop(columns=["TEAM_ID"], errors="ignore")

    merged = merged.merge(
        teams[["TEAM_ID", "NICKNAME"]],
        how="left",
        left_on="VISITOR_TEAM_ID",
        right_on="TEAM_ID",
    ).rename(columns={"NICKNAME": "AWAY_TEAM_NAME"}).drop(columns=["TEAM_ID"], errors="ignore")

    merged = merged.loc[:, ~merged.columns.duplicated()].copy()
    merged["PTS_HOME"] = pd.to_numeric(merged["PTS_HOME"], errors="coerce")
    merged["PTS_AWAY"] = pd.to_numeric(merged["PTS_AWAY"], errors="coerce")
    merged = merged.dropna(subset=["PTS_HOME", "PTS_AWAY"]).reset_index(drop=True)
    merged["HOME_WIN"] = (merged["PTS_HOME"] > merged["PTS_AWAY"]).astype(int)

    df_style = merged.copy()

    df_style["FG3_RATIO_HOME"] = df_style["FG3_PCT_HOME"] / (df_style["FG_PCT_HOME"] + 1e-6)
    df_style["FG3_RATIO_AWAY"] = df_style["FG3_PCT_AWAY"] / (df_style["FG_PCT_AWAY"] + 1e-6)

    if "AST_HOME" in df_style.columns and "TOV_HOME" in df_style.columns:
        df_style["AST_TOV_RATIO_HOME"] = df_style["AST_HOME"] / (df_style["TOV_HOME"] + 1e-6)
        df_style["AST_TOV_RATIO_AWAY"] = df_style["AST_AWAY"] / (df_style["TOV_AWAY"] + 1e-6)
    else:
        df_style["AST_TOV_RATIO_HOME"] = df_style["AST_HOME"]
        df_style["AST_TOV_RATIO_AWAY"] = df_style["AST_AWAY"]

    df_style["REB_MARGIN"] = df_style["REB_HOME"] - df_style["REB_AWAY"]
    df_style["FT_AGGRESSIVE_INDEX"] = df_style["FT_PCT_HOME"] - df_style["FT_PCT_AWAY"]
    df_style["PACE_INDEX"] = (df_style["PTS_HOME"] + df_style["PTS_AWAY"]) / 2

    style_features = [
        "FG3_RATIO_HOME",
        "AST_TOV_RATIO_HOME",
        "REB_MARGIN",
        "FT_AGGRESSIVE_INDEX",
        "PACE_INDEX"
    ]

    df_cluster = df_style.dropna(subset=style_features).copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_cluster[style_features])

    kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["PLAYSTYLE_CLUSTER"] = kmeans_final.fit_predict(X_scaled)

    df_style.loc[df_cluster.index, "PLAYSTYLE_CLUSTER"] = df_cluster["PLAYSTYLE_CLUSTER"]

    return df_style


def train_logistic_regression_model():
    df_style = load_and_prepare_data()

    feature_columns = [
        "FG3_RATIO_HOME",
        "AST_TOV_RATIO_HOME",
        "REB_MARGIN",
        "FT_AGGRESSIVE_INDEX",
        "PACE_INDEX",
    ]

    model_df = df_style.dropna(subset=feature_columns + ["HOME_WIN"]).copy()
    X = model_df[feature_columns]
    y = model_df["HOME_WIN"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_scaled, y_train)

    y_pred = logistic_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    return logistic_model, scaler, accuracy


def suggest_strategy_with_reason(playstyle_cluster, reb_margin, ast_tov_ratio):
    if reb_margin > 6:
        defense = "2-3 Zone Defense (focus on rebounding and rim protection)"
        reason_def = "A strong rebounding margin indicates control of the boards; a zone maximises that advantage and limits second-chance shots."
    elif playstyle_cluster == 0:
        defense = "Man-to-Man with perimeter containment (vs 3PT-heavy teams)"
        reason_def = "Opponent’s playstyle relies on perimeter spacing — man defense with tight close-outs reduces open 3s."
    elif playstyle_cluster == 1:
        defense = "Collapse-in zone (counter inside-focused play)"
        reason_def = "This cluster tends to attack inside; a compact zone helps crowd the paint and force outside shots."
    else:
        defense = "Switch defense / hybrid matchups (vs balanced offense)"
        reason_def = "A balanced opponent benefits from flexible matchups — switching helps neutralise screen-based sets."

    if ast_tov_ratio > 25:
        offense = "High ball movement (motion or pick-and-roll sets)"
        reason_off = "Excellent assist-to-turnover ratio — your team protects the ball well, so movement-based offense is recommended."
    elif reb_margin < -4:
        offense = "Attack in transition — exploit opponent rebounding gaps"
        reason_off = "A negative rebounding margin suggests a need to create easier scoring chances before the defense is set."
    else:
        offense = "Balanced tempo — emphasize spacing and drive-kick actions"
        reason_off = "The matchup does not point to one extreme, so balanced offense with spacing is the safest tactical recommendation."

    print("🏀 AI STRATEGY RECOMMENDATION")
    print("------------------------------------")
    print(f"Opponent Playstyle Cluster: {playstyle_cluster}")
    print(f"Rebounding Margin: {reb_margin:.2f}")
    print(f"Assist-to-Turnover Ratio: {ast_tov_ratio:.2f}")
    print()
    print(f"Defensive Recommendation: {defense}")
    print(f"Why: {reason_def}")
    print()
    print(f"Offensive Recommendation: {offense}")
    print(f"Why: {reason_off}")


def analyze_matchup(home_team, away_team):
    df_style = load_and_prepare_data()
    _logistic_model, _logistic_scaler, _logistic_accuracy = train_logistic_regression_model()

    numeric_cols = df_style.select_dtypes(include="number").columns

    home = df_style[df_style["HOME_TEAM_NAME"] == home_team][numeric_cols].mean()
    away = df_style[df_style["AWAY_TEAM_NAME"] == away_team][numeric_cols].mean()

    if home.empty or away.empty:
        print("⚠️ One or both team names not found. Please check spelling.")
        return

    reb_margin = home["REB_HOME"] - away["REB_AWAY"]
    ast_tov_ratio = home["AST_HOME"] / (home.get("TOV_HOME", 1) + 1e-6)
    playstyle_cluster = int(home["PLAYSTYLE_CLUSTER"])

    print(f"\n🏀 MATCHUP STRATEGY ANALYSIS: {home_team} (Home) vs {away_team} (Away)")
    suggest_strategy_with_reason(playstyle_cluster, reb_margin, ast_tov_ratio)


def generate_strategy(team_1, team_2):
    home_team = TEAM_NAME_MAP.get(team_1, team_1)
    away_team = TEAM_NAME_MAP.get(team_2, team_2)

    old_stdout = sys.stdout
    buffer = StringIO()
    sys.stdout = buffer

    try:
        analyze_matchup(home_team, away_team)
        output = buffer.getvalue().strip()
    finally:
        sys.stdout = old_stdout

    return output
