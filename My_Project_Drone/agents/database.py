"""
Phase 2 — Station Registry
PostgreSQL backend via psycopg2.

Credentials loaded from .env:
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

Distance: Python Haversine (PostGIS ST_Distance is a one-function swap if
           PostGIS extension is later installed via pg_application_stackbuilder).

Public API is identical to the old SQLite version — nothing else needs to change.
"""
from __future__ import annotations

import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Connection config from .env ───────────────────────────────────────────────
_DB_CFG = dict(
    host     = os.getenv("POSTGRES_HOST",     "localhost"),
    port     = int(os.getenv("POSTGRES_PORT", "5432")),
    dbname   = os.getenv("POSTGRES_DB",       "guardian_drone"),
    user     = os.getenv("POSTGRES_USER",     "postgres"),
    password = os.getenv("POSTGRES_PASSWORD", ""),
)

# 10 Hyderabad-area police stations (real area coordinates)
_SEED: list[tuple] = [
    (1,  "Hyderabad City Police HQ",  17.3850, 78.4867, 4),
    (2,  "Banjara Hills PS",          17.4239, 78.4492, 3),
    (3,  "Jubilee Hills PS",          17.4302, 78.4062, 2),
    (4,  "Gachibowli PS",             17.4435, 78.3476, 3),
    (5,  "Madhapur PS",               17.4490, 78.3878, 4),
    (6,  "Kondapur PS",               17.4672, 78.3523, 2),
    (7,  "HITEC City PS",             17.4476, 78.3779, 3),
    (8,  "Raidurg PS",                17.4334, 78.3683, 2),
    (9,  "Miyapur PS",                17.4967, 78.3578, 3),
    (10, "Kukatpally PS",             17.4849, 78.4053, 4),
]


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlam       = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@contextmanager
def _get_conn():
    """Yield a psycopg2 connection with RealDictCursor; auto-commit on success."""
    conn = psycopg2.connect(**_DB_CFG)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── Schema & seeding ─────────────────────────────────────────────────────────

def ensure_schema() -> None:
    """Create tables if they don't exist and seed station data."""
    with _get_conn() as conn:
        cur = conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                id            INTEGER PRIMARY KEY,
                name          TEXT    NOT NULL,
                lat           DOUBLE PRECISION NOT NULL,
                lon           DOUBLE PRECISION NOT NULL,
                drones_total  INTEGER NOT NULL DEFAULT 2,
                drones_avail  INTEGER NOT NULL DEFAULT 2
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS missions (
                id          TEXT PRIMARY KEY,
                station_id  INTEGER REFERENCES stations(id),
                drone_id    TEXT,
                status      TEXT        DEFAULT 'active',
                created_at  TIMESTAMPTZ DEFAULT NOW()
            )
        """)

        cur.execute("SELECT COUNT(*) FROM stations")
        if cur.fetchone()[0] == 0:
            cur.executemany(
                "INSERT INTO stations (id, name, lat, lon, drones_total, drones_avail) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [(s[0], s[1], s[2], s[3], s[4], s[4]) for s in _SEED],
            )


def reset_all_drones() -> None:
    """Restore all drones to full availability (for tests)."""
    ensure_schema()
    with _get_conn() as conn:
        conn.cursor().execute("UPDATE stations SET drones_avail = drones_total")


# ─── Public API ───────────────────────────────────────────────────────────────

class StationRow(NamedTuple):
    station_id:       int
    name:             str
    lat:              float
    lon:              float
    distance_km:      float
    drones_available: int


def nearest_stations(lat: float, lon: float, limit: int = 3) -> list[StationRow]:
    """
    Return up to `limit` nearest stations that have drones, sorted by distance.

    PostGIS upgrade path (when PostGIS extension is installed):
        SELECT id, name, lat, lon, drones_avail,
               ST_Distance(geom, ST_MakePoint(%s, %s)::geography) / 1000 AS dist_km
        FROM stations WHERE drones_avail > 0
        ORDER BY dist_km LIMIT %s
    """
    ensure_schema()
    with _get_conn() as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute("SELECT id, name, lat, lon, drones_avail FROM stations WHERE drones_avail > 0")
        rows = cur.fetchall()

    results = sorted(
        [
            StationRow(
                station_id       = r["id"],
                name             = r["name"],
                lat              = r["lat"],
                lon              = r["lon"],
                distance_km      = _haversine_km(lat, lon, r["lat"], r["lon"]),
                drones_available = r["drones_avail"],
            )
            for r in rows
        ],
        key=lambda s: s.distance_km,
    )
    return results[:limit]


def decrement_drone(station_id: int) -> bool:
    """Reserve one drone. Returns False if none available."""
    ensure_schema()
    with _get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE stations SET drones_avail = drones_avail - 1 "
            "WHERE id = %s AND drones_avail > 0",
            (station_id,),
        )
        return cur.rowcount == 1


def return_drone(station_id: int) -> None:
    """Release a drone back to the station (capped at drones_total)."""
    ensure_schema()
    with _get_conn() as conn:
        conn.cursor().execute(
            "UPDATE stations SET drones_avail = LEAST(drones_total, drones_avail + 1) "
            "WHERE id = %s",
            (station_id,),
        )


def log_mission(mission_id: str, station_id: int, drone_id: str) -> None:
    ensure_schema()
    with _get_conn() as conn:
        conn.cursor().execute(
            "INSERT INTO missions (id, station_id, drone_id) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET drone_id = EXCLUDED.drone_id",
            (mission_id, station_id, drone_id),
        )


def close_mission(mission_id: str) -> None:
    ensure_schema()
    with _get_conn() as conn:
        conn.cursor().execute(
            "UPDATE missions SET status = 'complete' WHERE id = %s",
            (mission_id,),
        )
