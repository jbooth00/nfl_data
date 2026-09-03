import json
import re
import nflreadpy as nfl
import polars as pl

def clean_player_name(name):
    if not name:
        return ""
    clean = re.sub(r'[^a-z0-9 ]', '', str(name).lower())
    return re.sub(r'\s+', ' ', clean).strip()

def fetch_player_stats_safe(target_season=2026):
    """
    Attempts to download weekly player stats for target_season.
    If the file returns 404 (e.g., season hasn't started or no data yet),
    it falls back to the previous season.
    """
    for season in [target_season, target_season - 1]:
        try:
            print(f"Attempting to fetch {season} weekly stats via nflreadpy...")
            df = nfl.load_player_stats([season])
            if not df.is_empty():
                print(f"Successfully loaded data for {season}!")
                return df, season
        except Exception as e:
            print(f"Notice: Could not load {season} data ({e}). Trying fallback...")
            
    return None, None

def build_nflverse_summary(target_season=2026, rolling_weeks=3):
    df, active_season = fetch_player_stats_safe(target_season)

    if df is None or df.is_empty():
        print("Error: Unable to retrieve player stats for target or fallback seasons.")
        return

    # Determine maximum week available in retrieved dataset
    max_week = df["week"].max()
    min_week = max(1, max_week - rolling_weeks + 1)
    
    print(f"Aggregating {active_season} data (Weeks {min_week} to {max_week})...")
    
    # Filter recent weeks
    df_recent = df.filter((pl.col("week") >= min_week) & (pl.col("week") <= max_week))

    # Aggregate key metrics per player
    summary = df_recent.group_by(["player_id", "player_name", "position"]).agg([
        pl.len().alias("games_sample"),
        pl.col("fantasy_points_ppr").mean().alias("actual_ppg"),
        pl.col("target_share").mean().alias("avg_target_share"),
        pl.col("air_yards_share").mean().alias("avg_air_yards_share"),
        # WOPR Formula = 1.5 * Target Share + 0.7 * Air Yards Share
        ((pl.col("target_share") * 1.5) + (pl.col("air_yards_share") * 0.7)).mean().alias("avg_wopr")
    ])

    output_data = {}
    for row in summary.iter_rows(named=True):
        c_name = clean_player_name(row["player_name"])
        if not c_name:
            continue
            
        output_data[c_name] = {
            "player_id": str(row["player_id"]),
            "raw_name": str(row["player_name"]),
            "position": str(row["position"]),
            "data_season": active_season,
            "games_sample": int(row["games_sample"]),
            "avg_target_share": round(float(row["avg_target_share"] or 0), 3),
            "avg_air_yards_share": round(float(row["avg_air_yards_share"] or 0), 3),
            "avg_wopr": round(float(row["avg_wopr"] or 0), 3),
            "actual_ppg": round(float(row["actual_ppg"] or 0), 2)
        }

    output_filename = "nflverse_summary.json"
    with open(output_filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Success! Exported {len(output_data)} player profiles to '{output_filename}'.")

if __name__ == "__main__":
    build_nflverse_summary(target_season=2026, rolling_weeks=3)