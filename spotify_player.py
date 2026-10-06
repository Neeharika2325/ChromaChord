"""Spotify search and browser playback helpers."""

import os
import threading
from collections import deque
from pathlib import Path

from dotenv import load_dotenv
from requests.exceptions import RequestException
from spotipy import Spotify
from spotipy.exceptions import SpotifyException
from spotipy.oauth2 import SpotifyClientCredentials


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

_RECENT_TRACK_LIMIT = 10
_recent_track_ids: deque[str] = deque(maxlen=_RECENT_TRACK_LIMIT)
_history_lock = threading.Lock()


def find_mood_track(search_query: str) -> tuple[str, str]:
    """Find a non-recent Spotify track and return its name and embed URL."""
    if not isinstance(search_query, str) or not search_query.strip():
        raise ValueError("search_query must be a non-empty string.")

    client_id = os.getenv("SPOTIPY_CLIENT_ID", "").strip()
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Spotify credentials are missing. Set SPOTIPY_CLIENT_ID and "
            "SPOTIPY_CLIENT_SECRET in the project's .env file."
        )

    try:
        spotify = Spotify(
            auth_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret,
            )
        )
        results = spotify.search(q=search_query.strip(), type="track", limit=10)
    except (SpotifyException, RequestException) as error:
        raise RuntimeError(f"Spotify search failed: {error}") from error

    tracks = results.get("tracks", {}).get("items", [])
    candidates = [
        track
        for track in tracks
        if track.get("id")
        and track.get("name")
        and track.get("artists")
        and track["artists"][0].get("name")
    ]
    if not candidates:
        raise LookupError(f"Spotify found no playable tracks for {search_query!r}.")

    with _history_lock:
        recent_ids = set(_recent_track_ids)
        track = next(
            (item for item in candidates if item["id"] not in recent_ids),
            candidates[0],
        )
        _recent_track_ids.append(track["id"])

    track_name = track["name"]
    artist_name = track["artists"][0]["name"]
    return (
        f"{track_name} - {artist_name}",
        f"https://open.spotify.com/embed/track/{track['id']}",
    )