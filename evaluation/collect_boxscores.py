"""
CoSQL NBA — Boxscore Data Collection
Purpose: Add boxscores and player_boxscores tables to nba_spatial DB
         for Craig/Sean's model evaluation. Additive only — does NOT
         touch shot_charts, games, players, or play_by_play.
Season:  2023-24 Boston Celtics (regular season + playoffs)
"""

import os
import time
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from dotenv import load_dotenv
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2

load_dotenv()

DB_HOST     = os.getenv("DB_HOST", "localhost")
DB_PORT     = os.getenv("DB_PORT", "5432")
DB_NAME     = os.getenv("DB_NAME", "nba_spatial")
DB_USER     = os.getenv("DB_USER", "rosalinatorres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

CELTICS_TEAM_ID = 1610612738
SEASON          = '2023-24'


CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS boxscores (
    id                  SERIAL PRIMARY KEY,
    game_id             TEXT,
    game_date           DATE,
    season              TEXT DEFAULT '2023-24',
    season_type         TEXT,        -- 'Regular' or 'Playoff'
    team_id             BIGINT,
    TEAM_ABBREVIATION   TEXT,
    PLAYER_ID           BIGINT,
    PLAYER_NAME         TEXT,
    PTS                 INTEGER,
    REB                 INTEGER,
    AST                 INTEGER,
    STL                 INTEGER,
    BLK                 INTEGER,
    TO_                 INTEGER,     -- turnovers (TO is reserved in SQL)
    FGM                 INTEGER,
    FGA                 INTEGER,
    FG3M                INTEGER,
    FG3A                INTEGER,
    FTM                 INTEGER,
    FTA                 INTEGER,
    MIN                 TEXT
);

CREATE TABLE IF NOT EXISTS player_boxscores (
    id                  SERIAL PRIMARY KEY,
    game_id             TEXT,
    game_date           DATE,
    season              TEXT DEFAULT '2023-24',
    season_type         TEXT,
    PLAYER_ID           BIGINT,
    PLAYER_NAME         TEXT,
    TEAM_ABBREVIATION   TEXT,
    PTS                 INTEGER,
    REB                 INTEGER,
    AST                 INTEGER,
    STL                 INTEGER,
    BLK                 INTEGER,
    FGM                 INTEGER,
    FGA                 INTEGER,
    FG3M                INTEGER,
    FG3A                INTEGER,
    FTM                 INTEGER,
    FTA                 INTEGER,
    MIN                 TEXT
);
"""


def connect():
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD
    )
    print(f"✅ Connected to {DB_NAME}")
    return conn


def fetch_celtics_games():
    print("📥 Fetching 2023-24 Celtics game list...")
    games = leaguegamefinder.LeagueGameFinder(
        season_nullable=SEASON,
        team_id_nullable=CELTICS_TEAM_ID
    ).get_data_frames()[0]

    games['GAME_ID'] = games['GAME_ID'].apply(lambda x: str(x).zfill(10))
    # Regular season (002) + Playoffs (004) only
    games = games[games['GAME_ID'].str.startswith(('002', '004'))].copy()
    games['season_type'] = games['GAME_ID'].apply(
        lambda x: 'Regular' if x.startswith('002') else 'Playoff'
    )
    print(f"  Found {len(games)} games ({len(games[games['season_type']=='Regular'])} regular, "
          f"{len(games[games['season_type']=='Playoff'])} playoff)")
    return games


def fetch_boxscores(games):
    print("📥 Fetching boxscores for each game...")
    all_rows = []

    for i, (_, game) in enumerate(games.iterrows()):
        game_id     = game['GAME_ID']
        game_date   = game.get('GAME_DATE')
        season_type = game['season_type']

        try:
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(
                game_id=game_id
            ).get_data_frames()[0]  # player stats frame

            for _, row in box.iterrows():
                all_rows.append({
                    'game_id':          game_id,
                    'game_date':        game_date,
                    'season_type':      season_type,
                    'team_id':          row.get('TEAM_ID'),
                    'TEAM_ABBREVIATION': row.get('TEAM_ABBREVIATION'),
                    'PLAYER_ID':        row.get('PLAYER_ID'),
                    'PLAYER_NAME':      row.get('PLAYER_NAME'),
                    'PTS':              row.get('PTS'),
                    'REB':              row.get('REB'),
                    'AST':              row.get('AST'),
                    'STL':              row.get('STL'),
                    'BLK':              row.get('BLK'),
                    'TO_':              row.get('TO'),
                    'FGM':              row.get('FGM'),
                    'FGA':              row.get('FGA'),
                    'FG3M':             row.get('FG3M'),
                    'FG3A':             row.get('FG3A'),
                    'FTM':              row.get('FTM'),
                    'FTA':              row.get('FTA'),
                    'MIN':              row.get('MIN'),
                })

            time.sleep(0.6)
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(games)} games...")

        except Exception as e:
            print(f"  ⚠️  Skipped game {game_id}: {e}")
            continue

    print(f"  Total player-game rows fetched: {len(all_rows)}")
    return pd.DataFrame(all_rows)


def to_int(val):
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return int(val)
    except (ValueError, TypeError):
        return None


def load(conn, df, games):
    cursor = conn.cursor()
    try:
        # Create tables
        cursor.execute(CREATE_TABLES)
        print("✅ Tables created (boxscores, player_boxscores)")

        # Build game_date lookup
        date_lookup = dict(zip(games['GAME_ID'], games['GAME_DATE']))

        # boxscores: all players from all games (both teams)
        boxscore_rows = [
            (row['game_id'], row['game_date'], '2023-24', row['season_type'],
             to_int(row['team_id']), row['TEAM_ABBREVIATION'],
             to_int(row['PLAYER_ID']), row['PLAYER_NAME'],
             to_int(row['PTS']), to_int(row['REB']), to_int(row['AST']),
             to_int(row['STL']), to_int(row['BLK']), to_int(row['TO_']),
             to_int(row['FGM']), to_int(row['FGA']),
             to_int(row['FG3M']), to_int(row['FG3A']),
             to_int(row['FTM']), to_int(row['FTA']),
             row['MIN'])
            for _, row in df.iterrows()
        ]
        execute_values(cursor, """
            INSERT INTO boxscores
                (game_id, game_date, season, season_type, team_id, TEAM_ABBREVIATION,
                 PLAYER_ID, PLAYER_NAME, PTS, REB, AST, STL, BLK, TO_, FGM, FGA,
                 FG3M, FG3A, FTM, FTA, MIN)
            VALUES %s
        """, boxscore_rows)
        print(f"  Loaded {len(boxscore_rows)} rows into boxscores")

        # player_boxscores: same data, slightly different shape (no team_id)
        player_rows = [
            (row['game_id'], row['game_date'], '2023-24', row['season_type'],
             to_int(row['PLAYER_ID']), row['PLAYER_NAME'], row['TEAM_ABBREVIATION'],
             to_int(row['PTS']), to_int(row['REB']), to_int(row['AST']),
             to_int(row['STL']), to_int(row['BLK']),
             to_int(row['FGM']), to_int(row['FGA']),
             to_int(row['FG3M']), to_int(row['FG3A']),
             to_int(row['FTM']), to_int(row['FTA']),
             row['MIN'])
            for _, row in df.iterrows()
        ]
        execute_values(cursor, """
            INSERT INTO player_boxscores
                (game_id, game_date, season, season_type, PLAYER_ID, PLAYER_NAME,
                 TEAM_ABBREVIATION, PTS, REB, AST, STL, BLK, FGM, FGA,
                 FG3M, FG3A, FTM, FTA, MIN)
            VALUES %s
        """, player_rows)
        print(f"  Loaded {len(player_rows)} rows into player_boxscores")

        conn.commit()

        # Verify
        cursor.execute("SELECT COUNT(*) FROM boxscores")
        print(f"\n✅ boxscores:        {cursor.fetchone()[0]} rows")
        cursor.execute("SELECT COUNT(*) FROM player_boxscores")
        print(f"✅ player_boxscores: {cursor.fetchone()[0]} rows")

    except Exception as e:
        conn.rollback()
        print(f"❌ Load failed: {e}")
        raise
    finally:
        cursor.close()


def main():
    print("=" * 60)
    print("CoSQL NBA — Boxscore Collection (2023-24 Celtics)")
    print("Additive only — existing tables untouched")
    print("=" * 60)

    conn = connect()
    try:
        games = fetch_celtics_games()
        df    = fetch_boxscores(games)
        load(conn, df, games)
        print("\n✅ DONE — boxscores and player_boxscores ready")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
