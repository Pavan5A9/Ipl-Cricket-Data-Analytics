"""
IPL Cricket Analytics Engine
Processes player and team statistics, calculates advanced cricket metrics,
generates professional insights, and exports styled charts and CSV reports.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Apply a professional styling theme
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16
})

def load_data(filepath: str) -> pd.DataFrame:
    """Loads IPL player performance dataset with error handling."""
    logging.info(f"Loading dataset from {filepath}...")
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
        
        ipl_df = pd.read_csv(filepath)
        logging.info(f"Successfully loaded dataset with {len(ipl_df)} records.")
        return ipl_df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def compute_advanced_metrics(ipl_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates advanced metrics such as Strike Rate for each row."""
    logging.info("Calculating player strike rates...")
    # Avoid division by zero by setting strike rate to 0 where balls faced is 0
    ipl_df["Strike_Rate"] = np.where(
        ipl_df["Balls"] > 0, 
        (ipl_df["Runs"] / ipl_df["Balls"]) * 100, 
        0.0
    ).round(2)
    return ipl_df

def analyze_players(ipl_df: pd.DataFrame) -> pd.DataFrame:
    """Computes career player statistics, including consistency metrics."""
    logging.info("Analyzing career performance metrics for players...")
    
    # Career aggregations
    player_stats = ipl_df.groupby("Player").agg(
        Matches=("Match_ID", "nunique"),
        Total_Runs=("Runs", "sum"),
        Total_Balls=("Balls", "sum"),
        Total_Wickets=("Wickets", "sum"),
        Total_Fours=("Fours", "sum"),
        Total_Sixes=("Sixes", "sum"),
        Mean_Runs=("Runs", "mean"),
        Std_Runs=("Runs", "std")
    ).reset_index()
    
    # Calculate Career Strike Rate and Batting Average
    player_stats["Strike_Rate"] = np.where(
        player_stats["Total_Balls"] > 0,
        (player_stats["Total_Runs"] / player_stats["Total_Balls"]) * 100,
        0.0
    ).round(2)
    
    player_stats["Batting_Average"] = (player_stats["Total_Runs"] / player_stats["Matches"]).round(2)
    
    # Consistency Index: Coefficient of Variation (CV) = StdDev / Mean
    # Lower CV indicates a more consistent player.
    player_stats["Consistency_Index"] = np.where(
        player_stats["Mean_Runs"] > 0,
        (player_stats["Std_Runs"] / player_stats["Mean_Runs"]) * 100,
        0.0
    ).round(2)
    
    # Sort by total runs
    player_stats = player_stats.sort_values(by="Total_Runs", ascending=False).reset_index(drop=True)
    return player_stats

def analyze_season_awards(ipl_df: pd.DataFrame) -> pd.DataFrame:
    """Identifies the Orange Cap and Purple Cap winners for each season."""
    logging.info("Analyzing season awards (Orange & Purple Caps)...")
    
    seasons = sorted(ipl_df["Season"].unique())
    awards = []
    
    for season in seasons:
        season_df = ipl_df[ipl_df["Season"] == season]
        
        # Orange Cap (Most Runs)
        runs_leader = season_df.groupby(["Player", "Team"])["Runs"].sum().idxmax()
        runs_val = season_df.groupby(["Player", "Team"])["Runs"].sum().max()
        
        # Purple Cap (Most Wickets)
        wickets_leader = season_df.groupby(["Player", "Team"])["Wickets"].sum().idxmax()
        wickets_val = season_df.groupby(["Player", "Team"])["Wickets"].sum().max()
        
        awards.append({
            "Season": season,
            "Orange_Cap_Player": runs_leader[0],
            "Orange_Cap_Team": runs_leader[1],
            "Orange_Cap_Runs": runs_val,
            "Purple_Cap_Player": wickets_leader[0],
            "Purple_Cap_Team": wickets_leader[1],
            "Purple_Cap_Wickets": wickets_val
        })
        
    return pd.DataFrame(awards)

def analyze_teams(ipl_df: pd.DataFrame) -> pd.DataFrame:
    """Computes overall team performance and win percentages."""
    logging.info("Analyzing team-level performance...")
    
    # Extract unique matches by team
    match_team_results = ipl_df.groupby(["Team", "Match_ID"])["Result"].first().reset_index()
    
    team_stats = match_team_results.groupby("Team").agg(
        Matches_Played=("Match_ID", "count"),
        Wins=("Result", lambda x: (x == "Win").sum()),
    ).reset_index()
    
    team_stats["Losses"] = team_stats["Matches_Played"] - team_stats["Wins"]
    team_stats["Win_Percentage"] = ((team_stats["Wins"] / team_stats["Matches_Played"]) * 100).round(2)
    
    # Get total runs scored by team
    team_runs = ipl_df.groupby("Team")["Runs"].sum().reset_index()
    team_stats = team_stats.merge(team_runs, on="Team")
    
    return team_stats.sort_values(by="Win_Percentage", ascending=False).reset_index(drop=True)

def analyze_toss_impact(ipl_df: pd.DataFrame) -> float:
    """Calculates the percentage of matches won by the team that won the toss."""
    logging.info("Analyzing toss decision impact...")
    
    # Unique matches and their details
    unique_matches = ipl_df.groupby("Match_ID").agg({
        "Toss_Winner": "first",
        "Team": "first",
        "Result": "first"
    }).reset_index()
    
    # Determine the winning team for each match
    # If Team A won, then Team A is Winner. If Team A lost, Team B (which is Opponent) is Winner.
    # In player performance, we can find the winning team by checking which team has Result == "Win"
    match_winners = ipl_df[ipl_df["Result"] == "Win"].groupby("Match_ID")["Team"].first().reset_index()
    match_winners.columns = ["Match_ID", "Match_Winner"]
    
    merged_matches = unique_matches.merge(match_winners, on="Match_ID")
    
    # Count how many times Toss Winner matches Match Winner
    toss_and_match_wins = (merged_matches["Toss_Winner"] == merged_matches["Match_Winner"]).sum()
    toss_win_pct = (toss_and_match_wins / len(merged_matches)) * 100
    
    return round(toss_win_pct, 2)

def generate_and_save_charts(ipl_df: pd.DataFrame, player_stats: pd.DataFrame, team_stats: pd.DataFrame, charts_dir: str) -> None:
    """Generates seven highly polished visualizations and saves them to charts_dir."""
    logging.info(f"Generating and exporting charts to {charts_dir}...")
    os.makedirs(charts_dir, exist_ok=True)
    
    # --- CHART 1: Career Runs vs Strike Rate (Scatter / Bubble) ---
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        data=player_stats, 
        x="Strike_Rate", 
        y="Batting_Average", 
        size="Total_Runs", 
        hue="Player",
        palette="viridis",
        sizes=(100, 600),
        alpha=0.85
    )
    plt.title("Player Career Profile: Strike Rate vs. Batting Average", pad=15)
    plt.xlabel("Career Strike Rate")
    plt.ylabel("Career Batting Average (Runs/Match)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Players (Bubble = Total Runs)")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "strike_rate_vs_average.png"), dpi=300)
    plt.close()
    
    # --- CHART 2: Team Win Percentage (Horizontal Bar) ---
    plt.figure(figsize=(9, 5))
    sorted_teams = team_stats.sort_values(by="Win_Percentage", ascending=True)
    colors = sns.color_palette("coolwarm", len(sorted_teams))
    bars = plt.barh(sorted_teams["Team"], sorted_teams["Win_Percentage"], color=colors, edgecolor="grey", height=0.6)
    
    # Add values on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%", 
                 va='center', ha='left', fontweight='bold', color='black')
                 
    plt.title("Overall Team Win Percentage (2021 - 2025)", pad=15)
    plt.xlabel("Win Percentage (%)")
    plt.ylabel("Team")
    plt.xlim(0, 100)
    plt.grid(True, axis='x', linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "team_wins_distribution.png"), dpi=300)
    plt.close()
    
    # --- CHART 3: Season-wise Runs Trend ---
    plt.figure(figsize=(9, 5))
    season_runs = ipl_df.groupby("Season")["Runs"].sum().reset_index()
    sns.lineplot(data=season_runs, x="Season", y="Runs", marker='o', color='#d95f02', linewidth=2.5, markersize=8)
    plt.title("Total Runs Scored Across Seasons", pad=15)
    plt.xlabel("Season")
    plt.ylabel("Total Runs")
    plt.xticks(season_runs["Season"])
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "season_runs_trend.png"), dpi=300)
    plt.close()

    # --- CHART 4: Toss Impact Visual (Pie Chart) ---
    plt.figure(figsize=(6, 6))
    toss_pct = analyze_toss_impact(ipl_df)
    labels = ['Toss Winner Won', 'Toss Winner Lost']
    sizes = [toss_pct, 100 - toss_pct]
    colors = ['#1b9e77', '#d95f02']
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors, 
            wedgeprops={"edgecolor":"black",'linewidth': 1, 'antialiased': True}, 
            textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title("Impact of Toss Decision on Match Outcome", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "toss_decision_impact.png"), dpi=300)
    plt.close()

    # --- CHART 5: Player Batting Performance Box Plot (Runs Distribution) ---
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=ipl_df,
        x="Player",
        y="Runs",
        palette="Set2"
    )
    plt.title("Distribution of Runs Scored by Key Players", pad=15)
    plt.xlabel("Player")
    plt.ylabel("Runs per Match")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y', linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "player_runs_distribution.png"), dpi=300)
    plt.close()

    # --- CHART 6: Boundaries Breakdown Stacked Bar Chart ---
    plt.figure(figsize=(10, 6))
    boundaries = ipl_df.groupby("Player").agg(
        Runs=("Runs", "sum"),
        Fours=("Fours", "sum"),
        Sixes=("Sixes", "sum")
    ).reset_index().sort_values(by="Runs", ascending=False)
    
    boundaries["Fours_Runs"] = boundaries["Fours"] * 4
    boundaries["Sixes_Runs"] = boundaries["Sixes"] * 6
    boundaries["Other_Runs"] = boundaries["Runs"] - boundaries["Fours_Runs"] - boundaries["Sixes_Runs"]
    
    players = boundaries["Player"]
    sixes_bars = boundaries["Sixes_Runs"]
    fours_bars = boundaries["Fours_Runs"]
    other_bars = boundaries["Other_Runs"]
    
    plt.bar(players, other_bars, label="Singles/Doubles", color="#a8dadc", edgecolor="grey")
    plt.bar(players, fours_bars, bottom=other_bars, label="Fours Runs (4s)", color="#457b9d", edgecolor="grey")
    plt.bar(players, sixes_bars, bottom=other_bars + fours_bars, label="Sixes Runs (6s)", color="#e63946", edgecolor="grey")
    
    plt.title("Run Scoring Breakdown: Boundaries vs Running (Career)", pad=15)
    plt.xlabel("Player")
    plt.ylabel("Total Runs Scored")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "boundaries_breakdown.png"), dpi=300)
    plt.close()

    # --- CHART 7: Performance Correlation Heatmap ---
    plt.figure(figsize=(8, 6))
    corr_df = ipl_df[["Runs", "Balls", "Wickets", "Fours", "Sixes", "Strike_Rate"]]
    corr_matrix = corr_df.corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        square=True
    )
    plt.title("Correlation Matrix of Player Performance Metrics", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "metrics_correlation_heatmap.png"), dpi=300)
    plt.close()


def generate_insights_report(player_stats: pd.DataFrame, team_stats: pd.DataFrame, awards_df: pd.DataFrame, toss_win_pct: float, filepath: str) -> None:
    """Generates a text summary of qualitative data-driven insights."""
    logging.info(f"Saving insights report to {filepath}...")
    
    top_scorer = player_stats.iloc[0]
    best_sr = player_stats.sort_values(by="Strike_Rate", ascending=False).iloc[0]
    most_consistent = player_stats.sort_values(by="Consistency_Index", ascending=True).iloc[0]
    best_team = team_stats.iloc[0]
    
    insights = f"""==========================================================
IPL CRICKET DATA ANALYTICS - INSIGHTS REPORT (2021-2025)
==========================================================

1. LEADING PERFORMERS (CAREER STATS):
   * Orange Cap Leader (Overall Runs): {top_scorer['Player']} ({top_scorer['Total_Runs']} Runs, Avg: {top_scorer['Batting_Average']})
   * Strike Rate King: {best_sr['Player']} (Strike Rate: {best_sr['Strike_Rate']}% over {best_sr['Matches']} matches)
   * Most Consistent Batter: {most_consistent['Player']} (Consistency Index (CV): {most_consistent['Consistency_Index']}%, lower is more reliable)

2. TEAM DOMINANCE:
   * Best Performing Team: {best_team['Team']} with a Win Percentage of {best_team['Win_Percentage']}% ({best_team['Wins']} Wins out of {best_team['Matches_Played']} Matches).

3. TOSS ADVANTAGE ANALYSIS:
   * Teams winning the toss went on to win the match {toss_win_pct}% of the time. This indicates a {'moderate' if toss_win_pct > 50 else 'negligible'} home/toss advantage in the tournament conditions.

4. SEASON AWARDS SUMMARY:
"""
    for _, row in awards_df.iterrows():
        insights += f"   * Season {row['Season']}: Orange Cap: {row['Orange_Cap_Player']} ({row['Orange_Cap_Runs']} Runs) | Purple Cap: {row['Purple_Cap_Player']} ({row['Purple_Cap_Wickets']} Wickets)\n"
        
    insights += "\nProject Completed Successfully."
    
    print("\n" + insights)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(insights)

def main():
    # Setup directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    data_path = os.path.join(project_root, "data", "ipl_player_performance.csv")
    charts_dir = os.path.join(project_root, "charts")
    reports_dir = os.path.join(project_root, "reports")
    
    # 1. Load Data
    ipl_df = load_data(data_path)
    
    # 2. Process metrics
    ipl_df = compute_advanced_metrics(ipl_df)
    
    # 3. Calculate statistics
    player_stats = analyze_players(ipl_df)
    awards_df = analyze_season_awards(ipl_df)
    team_stats = analyze_teams(ipl_df)
    toss_pct = analyze_toss_impact(ipl_df)
    
    # 4. Save CSV Reports
    os.makedirs(reports_dir, exist_ok=True)
    player_stats.to_csv(os.path.join(reports_dir, "ipl_overall_analysis_report.csv"), index=False)
    team_stats.to_csv(os.path.join(reports_dir, "ipl_team_summary.csv"), index=False)
    awards_df.to_csv(os.path.join(reports_dir, "ipl_season_awards.csv"), index=False)
    logging.info("Saved CSV reports successfully.")
    
    # 5. Generate and Save Charts
    generate_and_save_charts(ipl_df, player_stats, team_stats, charts_dir)
    
    # 6. Generate Insights and Print
    insights_path = os.path.join(reports_dir, "ipl_insights_report.txt")
    generate_insights_report(player_stats, team_stats, awards_df, toss_pct, insights_path)

if __name__ == "__main__":
    main()
