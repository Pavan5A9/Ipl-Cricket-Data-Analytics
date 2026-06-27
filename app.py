"""
IPL Cricket Analytics - Streamlit Dashboard
An interactive web application showcasing key performance indicators,
advanced player comparison tools, and dynamic Plotly charts.
"""

import streamlit as pd_st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration
pd_st.set_page_config(
    page_title="IPL Cricket Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium styling
pd_st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border: 1px solid #e9ecef;
        text-align: center;
    }
    .metric-title {
        font-size: 14px;
        color: #6c757d;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        color: #1d3557;
        font-weight: 700;
    }
    .metric-subtitle {
        font-size: 12px;
        color: #a8dadc;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@pd_st.cache_data
def load_dashboard_data(filepath):
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath)
    df["Strike_Rate"] = np.where(df["Balls"] > 0, (df["Runs"] / df["Balls"]) * 100, 0.0).round(2)
    return df

# Get filepath
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
data_path = os.path.join(project_root, "data", "ipl_player_performance.csv")

df = load_dashboard_data(data_path)

if df is None:
    pd_st.error(f"Dataset not found at `{data_path}`. Please run `src/data_generator.py` first.")
else:
    # Sidebar Filters
    pd_st.sidebar.title("🏏 Filter Dashboard")
    pd_st.sidebar.markdown("Use filters below to customize the visualizations and metrics.")
    
    # Season Filter
    seasons = sorted(df["Season"].unique())
    selected_season = pd_st.sidebar.selectbox("Select Season", ["All Seasons"] + list(seasons))
    
    # Team Filter
    teams = sorted(df["Team"].unique())
    selected_team = pd_st.sidebar.selectbox("Select Team", ["All Teams"] + list(teams))
    
    # Filter dataset
    filtered_df = df.copy()
    if selected_season != "All Seasons":
        filtered_df = filtered_df[filtered_df["Season"] == selected_season]
    if selected_team != "All Teams":
        filtered_df = filtered_df[(filtered_df["Team"] == selected_team) | (filtered_df["Opponent"] == selected_team)]
        
    # Title Section
    pd_st.title("🏏 IPL Cricket Performance Analytics")
    pd_st.markdown("A premium, interactive data dashboard analyzing player performances, match results, and team statistics from 2021 to 2025.")
    pd_st.markdown("---")
    
    # Calculate Overall Metrics for KPI Cards
    # 1. Orange Cap Leader
    player_runs = filtered_df.groupby("Player")["Runs"].sum().reset_index()
    if not player_runs.empty:
        top_runs_idx = player_runs["Runs"].idxmax()
        top_runs_player = player_runs.loc[top_runs_idx, "Player"]
        top_runs_value = player_runs.loc[top_runs_idx, "Runs"]
    else:
        top_runs_player, top_runs_value = "N/A", 0
        
    # 2. Purple Cap Leader
    player_wkts = filtered_df.groupby("Player")["Wickets"].sum().reset_index()
    if not player_wkts.empty:
        top_wkts_idx = player_wkts["Wickets"].idxmax()
        top_wkts_player = player_wkts.loc[top_wkts_idx, "Player"]
        top_wkts_value = player_wkts.loc[top_wkts_idx, "Wickets"]
    else:
        top_wkts_player, top_wkts_value = "N/A", 0
        
    # 3. Best Strike Rate (Min 50 runs)
    player_sr = filtered_df.groupby("Player").agg(
        Runs=("Runs", "sum"),
        Balls=("Balls", "sum")
    ).reset_index()
    player_sr_qualified = player_sr[player_sr["Runs"] >= 50]
    if not player_sr_qualified.empty:
        player_sr_qualified["SR"] = (player_sr_qualified["Runs"] / player_sr_qualified["Balls"] * 100).round(2)
        top_sr_idx = player_sr_qualified["SR"].idxmax()
        top_sr_player = player_sr_qualified.loc[top_sr_idx, "Player"]
        top_sr_value = player_sr_qualified.loc[top_sr_idx, "SR"]
    else:
        top_sr_player, top_sr_value = "N/A", 0.0

    # 4. Best Team Win Rate
    # Unique match records per team
    match_results = filtered_df.groupby(["Team", "Match_ID"])["Result"].first().reset_index()
    team_stats = match_results.groupby("Team").agg(
        Played=("Match_ID", "count"),
        Wins=("Result", lambda x: (x == "Win").sum())
    ).reset_index()
    if not team_stats.empty:
        team_stats["WinRate"] = (team_stats["Wins"] / team_stats["Played"] * 100).round(2)
        top_team_idx = team_stats["WinRate"].idxmax()
        top_team_name = team_stats.loc[top_team_idx, "Team"]
        top_team_value = team_stats.loc[top_team_idx, "WinRate"]
    else:
        top_team_name, top_team_value = "N/A", 0.0
        
    # Display KPI Cards
    col1, col2, col3, col4 = pd_st.columns(4)
    with col1:
        pd_st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">🏆 ORANGE CAP LEADER</div>
            <div class="metric-value">{top_runs_value} Runs</div>
            <div style="color: #e63946; font-weight: bold; font-size: 14px;">{top_runs_player}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        pd_st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">💜 PURPLE CAP LEADER</div>
            <div class="metric-value">{top_wkts_value} Wkts</div>
            <div style="color: #8338ec; font-weight: bold; font-size: 14px;">{top_wkts_player}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        pd_st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⚡ HIGHEST STRIKE RATE</div>
            <div class="metric-value">{top_sr_value}%</div>
            <div style="color: #ffb703; font-weight: bold; font-size: 14px;">{top_sr_player} (Min 50 Runs)</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        pd_st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">⭐ DOMINANT TEAM</div>
            <div class="metric-value">{top_team_value}% Wins</div>
            <div style="color: #457b9d; font-weight: bold; font-size: 14px;">{top_team_name}</div>
        </div>
        """, unsafe_allow_html=True)

    pd_st.markdown("<br>", unsafe_allow_html=True)

    # Interactive Plotly Visualizations Section
    pd_st.header("📊 Interactive Performance Visualizations")
    
    col_chart1, col_chart2 = pd_st.columns(2)
    
    with col_chart1:
        # Chart 1: Batting Profile Scatter (Average vs Strike Rate)
        player_agg = filtered_df.groupby("Player").agg(
            Runs=("Runs", "sum"),
            Balls=("Balls", "sum"),
            Sixes=("Sixes", "sum"),
            Matches=("Match_ID", "nunique"),
            Team=("Team", "first")
        ).reset_index()
        player_agg["Average"] = (player_agg["Runs"] / player_agg["Matches"]).round(2)
        player_agg["SR"] = np.where(player_agg["Balls"] > 0, (player_agg["Runs"] / player_agg["Balls"] * 100), 0.0).round(2)
        
        fig1 = px.scatter(
            player_agg,
            x="SR",
            y="Average",
            size="Runs",
            color="Team",
            hover_name="Player",
            labels={"SR": "Strike Rate (%)", "Average": "Batting Average (Runs/Innings)"},
            title="Player Career Profile: Strike Rate vs Batting Average",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig1.update_layout(legend_title="Team (Bubble = Total Runs)")
        pd_st.plotly_chart(fig1, use_container_width=True)
        
    with col_chart2:
        # Chart 2: Team Win Percentages
        # Unique match details for results bar chart
        match_teams_filtered = filtered_df.groupby(["Team", "Match_ID"])["Result"].first().reset_index()
        team_win_data = match_teams_filtered.groupby("Team").agg(
            Played=("Match_ID", "count"),
            Wins=("Result", lambda x: (x == "Win").sum())
        ).reset_index()
        team_win_data["WinRate"] = (team_win_data["Wins"] / team_win_data["Played"] * 100).round(2)
        team_win_data = team_win_data.sort_values(by="WinRate", ascending=False)
        
        fig2 = px.bar(
            team_win_data,
            x="Team",
            y="WinRate",
            color="WinRate",
            labels={"WinRate": "Win Percentage (%)", "Team": "IPL Franchise"},
            title="Team Success Rate Summary",
            color_continuous_scale="Viridis",
            text_auto=".1f"
        )
        fig2.update_layout(coloraxis_showscale=False)
        pd_st.plotly_chart(fig2, use_container_width=True)

    col_chart3, col_chart4 = pd_st.columns(2)
    
    with col_chart3:
        # Chart 3: Season Run comparison
        season_comparison = filtered_df.groupby(["Season", "Team"])["Runs"].sum().reset_index()
        fig3 = px.bar(
            season_comparison,
            x="Season",
            y="Runs",
            color="Team",
            barmode="group",
            title=" франшиза Season-wise Runs Accumulation",
            labels={"Runs": "Total Runs Scored"},
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        pd_st.plotly_chart(fig3, use_container_width=True)
        
    with col_chart4:
        # Chart 4: Toss Impact Analysis (Pie)
        # Using unique matches from selected data
        unique_matches_filter = filtered_df.groupby("Match_ID").first().reset_index()
        
        # Match Winners list
        match_winners_f = filtered_df[filtered_df["Result"] == "Win"].groupby("Match_ID")["Team"].first().reset_index()
        match_winners_f.columns = ["Match_ID", "Winner"]
        
        merged_toss = unique_matches_filter.merge(match_winners_f, on="Match_ID")
        merged_toss["TossOutcome"] = np.where(merged_toss["Toss_Winner"] == merged_toss["Winner"], "Toss Winner Won", "Toss Winner Lost")
        toss_counts = merged_toss["TossOutcome"].value_counts().reset_index()
        toss_counts.columns = ["Outcome", "Count"]
        
        fig4 = px.pie(
            toss_counts,
            names="Outcome",
            values="Count",
            title="Toss Advantage Impact on Match Outcome",
            color="Outcome",
            color_discrete_map={"Toss Winner Won": "#2a9d8f", "Toss Winner Lost": "#e76f51"}
        )
        pd_st.plotly_chart(fig4, use_container_width=True)

    col_chart5, col_chart6 = pd_st.columns(2)
    
    with col_chart5:
        # Chart 5: Runs Distribution (Box Plot)
        fig5 = px.box(
            filtered_df,
            x="Player",
            y="Runs",
            color="Player",
            title="Runs Distribution per Innings by Player",
            points="outliers",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig5.update_layout(xaxis_tickangle=-45)
        pd_st.plotly_chart(fig5, use_container_width=True)
        
    with col_chart6:
        # Chart 6: Boundaries Breakdown Stacked Bar Chart
        df_boundaries = filtered_df.groupby("Player").agg(
            Fours=("Fours", "sum"),
            Sixes=("Sixes", "sum"),
            Runs=("Runs", "sum")
        ).reset_index().sort_values(by="Runs", ascending=False)
        df_boundaries["Fours_Runs"] = df_boundaries["Fours"] * 4
        df_boundaries["Sixes_Runs"] = df_boundaries["Sixes"] * 6
        df_boundaries["Other_Runs"] = df_boundaries["Runs"] - df_boundaries["Fours_Runs"] - df_boundaries["Sixes_Runs"]
        
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(
            x=df_boundaries["Player"],
            y=df_boundaries["Other_Runs"],
            name="Singles/Doubles",
            marker_color="#a8dadc"
        ))
        fig6.add_trace(go.Bar(
            x=df_boundaries["Player"],
            y=df_boundaries["Fours_Runs"],
            name="Fours Runs (4s)",
            marker_color="#457b9d"
        ))
        fig6.add_trace(go.Bar(
            x=df_boundaries["Player"],
            y=df_boundaries["Sixes_Runs"],
            name="Sixes Runs (6s)",
            marker_color="#e63946"
        ))
        fig6.update_layout(
            barmode='stack',
            title="Run Scoring Breakdown: Boundaries vs Running",
            xaxis_title="Player",
            yaxis_title="Total Runs",
            legend_title="Scoring Type",
            xaxis_tickangle=-45
        )
        pd_st.plotly_chart(fig6, use_container_width=True)

    pd_st.markdown("---")

    # Player Comparison Side-by-Side Tool
    pd_st.header("👥 Player Head-to-Head Comparison")
    pd_st.markdown("Compare key career batting and bowling statistics for any two players side-by-side.")
    
    all_players = sorted(df["Player"].unique())
    col_p1, col_p2 = pd_st.columns(2)
    
    with col_p1:
        player_a = pd_st.selectbox("Select Player A", all_players, index=0)
    with col_p2:
        player_b = pd_st.selectbox("Select Player B", all_players, index=1 if len(all_players) > 1 else 0)
        
    # Get stats for selected players
    # Aggregated from the overall dataset for career stats
    career_stats = df.groupby("Player").agg(
        Matches=("Match_ID", "nunique"),
        Runs=("Runs", "sum"),
        Balls=("Balls", "sum"),
        Wickets=("Wickets", "sum"),
        Fours=("Fours", "sum"),
        Sixes=("Sixes", "sum")
    ).reset_index()
    career_stats["Avg"] = (career_stats["Runs"] / career_stats["Matches"]).round(2)
    career_stats["SR"] = np.where(career_stats["Balls"] > 0, (career_stats["Runs"] / career_stats["Balls"] * 100), 0.0).round(2)
    
    stats_a = career_stats[career_stats["Player"] == player_a].iloc[0]
    stats_b = career_stats[career_stats["Player"] == player_b].iloc[0]
    
    # Comparison table layout
    comp_df = pd.DataFrame({
        "Metric": ["Matches Played", "Total Runs", "Batting Average", "Strike Rate (%)", "Total Wickets", "Fours Hit", "Sixes Hit"],
        player_a: [
            int(stats_a["Matches"]), 
            int(stats_a["Runs"]), 
            stats_a["Avg"], 
            stats_a["SR"], 
            int(stats_a["Wickets"]), 
            int(stats_a["Fours"]), 
            int(stats_a["Sixes"])
        ],
        player_b: [
            int(stats_b["Matches"]), 
            int(stats_b["Runs"]), 
            stats_b["Avg"], 
            stats_b["SR"], 
            int(stats_b["Wickets"]), 
            int(stats_b["Fours"]), 
            int(stats_b["Sixes"])
        ]
    })
    
    # Display radar chart and table side by side
    col_comp_radar, col_comp_table = pd_st.columns([1, 1])
    
    with col_comp_radar:
        categories = ['Runs', 'Batting Avg', 'Strike Rate', 'Wickets', 'Boundaries']
        
        # Max scaling factors based on career stats
        max_runs = max(career_stats["Runs"].max(), 1)
        max_avg = max(career_stats["Avg"].max(), 1)
        max_sr = max(career_stats["SR"].max(), 1)
        max_wkts = max(career_stats["Wickets"].max(), 1)
        max_bounds = max((career_stats["Fours"] + career_stats["Sixes"]).max(), 1)
        
        val_a = [
            (stats_a["Runs"] / max_runs) * 100,
            (stats_a["Avg"] / max_avg) * 100,
            (stats_a["SR"] / max_sr) * 100,
            (stats_a["Wickets"] / max_wkts) * 100,
            ((stats_a["Fours"] + stats_a["Sixes"]) / max_bounds) * 100
        ]
        val_b = [
            (stats_b["Runs"] / max_runs) * 100,
            (stats_b["Avg"] / max_avg) * 100,
            (stats_b["SR"] / max_sr) * 100,
            (stats_b["Wickets"] / max_wkts) * 100,
            ((stats_b["Fours"] + stats_b["Sixes"]) / max_bounds) * 100
        ]
        
        # Close the loop
        val_a_close = val_a + [val_a[0]]
        val_b_close = val_b + [val_b[0]]
        categories_close = categories + [categories[0]]
        
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=val_a_close,
            theta=categories_close,
            fill='toself',
            name=player_a,
            fillcolor='rgba(69, 123, 157, 0.4)',
            line=dict(color='#457b9d')
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=val_b_close,
            theta=categories_close,
            fill='toself',
            name=player_b,
            fillcolor='rgba(230, 57, 70, 0.4)',
            line=dict(color='#e63946')
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            title=dict(
                text="Player Career Metric Comparison (Normalized %)",
                x=0.5,
                xanchor='center'
            ),
            showlegend=True
        )
        pd_st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_comp_table:
        pd_st.markdown("<br><br>", unsafe_allow_html=True)
        pd_st.table(comp_df.set_index("Metric"))

    pd_st.markdown("---")

    # Data Explorer
    pd_st.header("🔍 Dataset Explorer")
    pd_st.markdown("View or filter the raw player performances records.")
    
    search_player = pd_st.text_input("Search Player by Name", "")
    
    explorer_df = df.copy()
    if search_player:
        explorer_df = explorer_df[explorer_df["Player"].str.contains(search_player, case=False, na=False)]
        
    pd_st.dataframe(explorer_df, use_container_width=True)
    
    # Download Button
    csv = explorer_df.to_csv(index=False).encode('utf-8')
    pd_st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="ipl_filtered_data.csv",
        mime="text/csv"
    )
