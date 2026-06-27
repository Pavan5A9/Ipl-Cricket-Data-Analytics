"""
IPL Cricket Data Generator
Generates a realistic, multi-season synthetic dataset representing match-by-match
performances of key IPL players. This data is used to drive the analytics pipeline
and interactive dashboard.
"""

import os
import logging
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def generate_dataset(output_path: str, seed: int = 42) -> None:
    """
    Generates a realistic synthetic IPL player performance dataset and saves it as a CSV.
    
    Parameters:
        output_path (str): Filepath where the CSV should be saved.
        seed (int): Random seed for reproducibility.
    """
    np.random.seed(seed)
    logging.info("Starting synthetic dataset generation...")

    # Configuration for Players and Profiles
    # Format: (Player, Team, Role: batsman/bowler/allrounder, base_avg_runs, base_sr, base_wickets)
    player_profiles = [
        ("Virat Kohli", "RCB", "batsman", 48, 132.5, 0.02),
        ("Rohit Sharma", "MI", "batsman", 32, 138.0, 0.01),
        ("MS Dhoni", "CSK", "batsman", 26, 155.0, 0.0),
        ("Shubman Gill", "GT", "batsman", 44, 135.5, 0.0),
        ("Suryakumar Yadav", "MI", "batsman", 38, 168.0, 0.0),
        ("KL Rahul", "LSG", "batsman", 46, 128.0, 0.0),
        ("Ravindra Jadeja", "CSK", "allrounder", 22, 142.0, 1.1),
        ("Hardik Pandya", "MI", "allrounder", 28, 146.5, 0.9),
        ("Rashid Khan", "GT", "bowler", 12, 150.0, 1.4),
        ("Jasprit Bumrah", "MI", "bowler", 4, 105.0, 1.6),
        ("Rinku Singh", "KKR", "batsman", 30, 150.0, 0.0),
        ("Yuzvendra Chahal", "RR", "bowler", 2, 80.0, 1.5)
    ]
    
    teams = list(set(profile[1] for profile in player_profiles))
    venues = [
        "Wankhede Stadium, Mumbai", 
        "M. Chinnaswamy Stadium, Bengaluru", 
        "MA Chidambaram Stadium, Chennai", 
        "Narendra Modi Stadium, Ahmedabad", 
        "Eden Gardens, Kolkata"
    ]
    
    seasons = [2021, 2022, 2023, 2024, 2025]
    matches_per_season = 25
    
    data_rows = []
    match_counter = 1
    
    for season in seasons:
        for match_idx in range(1, matches_per_season + 1):
            # Select two random teams for this match
            team_a, team_b = np.random.choice(teams, size=2, replace=False)
            venue = np.random.choice(venues)
            
            # Toss logic
            toss_winner = np.random.choice([team_a, team_b])
            toss_decision = np.random.choice(["Bat", "Bowl"])
            
            # Match outcome
            # Give a small advantage to toss winner or home team (simulated randomly)
            result_prob = 0.52 if toss_winner == team_a else 0.48
            team_a_won = np.random.random() < result_prob
            
            # Find players participating in this match from team_a and team_b
            match_players = [p for p in player_profiles if p[1] in [team_a, team_b]]
            
            for player, team, role, base_runs, base_sr, base_wkt in match_players:
                opponent = team_b if team == team_a else team_a
                is_win = team_a_won if team == team_a else not team_a_won
                result = "Win" if is_win else "Lose"
                
                # Performance generation based on player profile
                if role == "batsman":
                    # Batsman runs: Log-normal or normal distribution
                    runs = int(np.clip(np.random.normal(base_runs, base_runs * 0.6), 0, 125))
                    # Strike rate logic: varies around base
                    sr = np.clip(np.random.normal(base_sr, 15), 80, 250)
                    balls = int(round((runs / sr) * 100)) if runs > 0 else 0
                    if runs > 0 and balls == 0:
                        balls = 1
                    wickets = int(np.random.poisson(base_wkt))
                    
                elif role == "allrounder":
                    # Allrounder gets moderate batting + bowling stats
                    runs = int(np.clip(np.random.normal(base_runs, base_runs * 0.8), 0, 75))
                    sr = np.clip(np.random.normal(base_sr, 20), 70, 220)
                    balls = int(round((runs / sr) * 100)) if runs > 0 else 0
                    if runs > 0 and balls == 0:
                        balls = 1
                    wickets = int(np.random.poisson(base_wkt))
                    
                else:  # bowler
                    # Bowler gets low batting runs, but high wickets
                    runs = int(np.clip(np.random.exponential(base_runs), 0, 30))
                    if runs > 0:
                        sr = np.clip(np.random.normal(base_sr, 25), 50, 180)
                        balls = int(round((runs / sr) * 100))
                        if balls == 0:
                            balls = 1
                    else:
                        balls = 0
                    wickets = int(np.random.poisson(base_wkt))
                
                # Fours and Sixes logic
                fours = 0
                sixes = 0
                if runs > 0:
                    # Distribute runs into boundaries based on strike rate
                    boundary_run_pct = np.clip(0.3 + (sr - 100) / 300, 0.1, 0.8)
                    boundary_runs = runs * boundary_run_pct
                    
                    # Assume mix of 4s and 6s (e.g. MS Dhoni or Surya hits more 6s)
                    six_ratio = 0.4 if "Dhoni" in player or "Surya" in player or "Pandya" in player else 0.25
                    
                    sixes = int(np.clip((boundary_runs * six_ratio) // 6, 0, runs // 6))
                    remaining_boundary = boundary_runs - (sixes * 6)
                    fours = int(np.clip(remaining_boundary // 4, 0, (runs - sixes * 6) // 4))
                
                # Append row
                data_rows.append({
                    "Match_ID": f"M_{match_counter:04d}",
                    "Player": player,
                    "Team": team,
                    "Opponent": opponent,
                    "Runs": runs,
                    "Balls": balls,
                    "Wickets": wickets,
                    "Fours": fours,
                    "Sixes": sixes,
                    "Season": season,
                    "Venue": venue,
                    "Toss_Winner": toss_winner,
                    "Toss_Decision": toss_decision,
                    "Result": result
                })
            
            match_counter += 1

    df = pd.DataFrame(data_rows)
    
    # Recalculate strike rate for validation
    df["Strike_Rate"] = np.where(df["Balls"] > 0, (df["Runs"] / df["Balls"]) * 100, 0.0).round(2)
    
    # Save directory verification
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Successfully generated dataset with {len(df)} rows. Saved to: {output_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_file_path = os.path.join(project_root, "data", "ipl_player_performance.csv")
    
    generate_dataset(data_file_path)
