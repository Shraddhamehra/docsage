"""Database connection + schema setup."""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://localhost:5432/docsage")

SCHEMA_FILE = Path(__file__).parent.parent / "schema.sql"


def get_conn() -> psycopg.Connection:
    """Open a new database connection."""
    return psycopg.connect(DATABASE_URL)


def init_schema() -> None:
    """Create tables and indexes if they don't exist yet."""
    with get_conn() as conn:
        conn.execute(SCHEMA_FILE.read_text())


if __name__ == "__main__":
    init_schema()
    print("schema ready")
