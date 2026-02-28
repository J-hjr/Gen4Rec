"""
Build SQLite database from local Music4All folder.

Expected repo structure:

GEN4REC/
├── music4all/
│   ├── id_information.csv
│   ├── id_genres.csv
│   ├── id_lang.csv
│   ├── id_metadata.csv
│   ├── id_tags.csv
│   └── listening_history.csv
│
└── src/data/build_music4all_db.py
"""

import os
import sqlite3
import pandas as pd
from typing import Dict

# -----------------------------
# Paths (repo-relative)
# -----------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MUSIC4ALL_DIR = os.path.join(REPO_ROOT, "music4all")
DB_PATH = os.path.join(REPO_ROOT, "music4all.db")

# -----------------------------
# Tables to load
# -----------------------------

TABLES: Dict[str, str] = {
    "id_information": "id_information.csv",
    "id_genres": "id_genres.csv",
    "id_lang": "id_lang.csv",
    "id_metadata": "id_metadata.csv",
    "id_tags": "id_tags.csv",
    "listening_history": "listening_history.csv",
}

# -----------------------------
# Helper: Smart CSV reader
# -----------------------------

def read_csv_safely(path: str, chunksize=None):
    """
    Some Music4All .csv files are actually tab-separated.
    Try tab first, then fallback to comma.
    """
    try:
        return pd.read_csv(path, sep="\t", on_bad_lines="skip", chunksize=chunksize)
    except Exception:
        return pd.read_csv(path, sep=",", on_bad_lines="skip", chunksize=chunksize)

# -----------------------------
# Main DB builder
# -----------------------------

def build_database():

    if not os.path.exists(MUSIC4ALL_DIR):
        raise FileNotFoundError(f"music4all folder not found at: {MUSIC4ALL_DIR}")

    print(f"[INFO] Building database at: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for table_name, filename in TABLES.items():
        file_path = os.path.join(MUSIC4ALL_DIR, filename)

        if not os.path.exists(file_path):
            print(f"[WARNING] Missing file: {filename}")
            continue

        print(f"[LOAD] {filename} → {table_name}")

        # Large table handling
        if table_name == "listening_history":
            reader = read_csv_safely(file_path, chunksize=200_000)
            for i, chunk in enumerate(reader):
                mode = "replace" if i == 0 else "append"
                chunk.to_sql(table_name, conn, if_exists=mode, index=False)
        else:
            df = read_csv_safely(file_path)
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        print("   ✓ Done")

    # -----------------------------
    # Create useful indices
    # -----------------------------

    print("[INFO] Creating indices...")

    for table_name in TABLES.keys():
        try:
            columns = pd.read_sql_query(
                f"PRAGMA table_info({table_name});",
                conn
            )["name"].tolist()
        except Exception:
            continue

        if "id" in columns:
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_id "
                f"ON {table_name}(id);"
            )

        if table_name == "listening_history":
            for col in ["user_id", "track_id", "song_id"]:
                if col in columns:
                    cursor.execute(
                        f"CREATE INDEX IF NOT EXISTS idx_{table_name}_{col} "
                        f"ON {table_name}({col});"
                    )

    conn.commit()
    conn.close()

    print("[SUCCESS] Database ready.")

# -----------------------------
# CLI Entry
# -----------------------------

if __name__ == "__main__":
    build_database()