#!/usr/bin/env python3
"""Utility script to call the dungeon API and start a game with a custom map string."""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Optional

import requests


def load_map_text(map_path: pathlib.Path) -> str:
    """Read the map text from disk, raising a clear error when missing."""
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")
    data = map_path.read_text(encoding="utf-8")
    cleaned = data.strip()
    if not cleaned:
        raise ValueError(f"Map file {map_path} is empty")
    return cleaned


def start_game(host: str, port: int, map_text: str) -> dict:
    """Send a start_game request with the provided map text."""
    url = f"http://{host}:{port}/start_game"
    payload = {
        "mode": "string",
        "custom_map": map_text,
    }
    resp = requests.post(url, json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    default_map = pathlib.Path("data/map_dataset/01_navigation/single_turn_15.txt")

    parser = argparse.ArgumentParser(description="Start Dungeon Escape game with a custom map text.")
    parser.add_argument(
        "map",
        nargs="?",
        default=default_map,
        type=pathlib.Path,
        help=f"Path to the .txt file containing map layout (default: {default_map})",
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Dungeon API host (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Dungeon API port (default: 8000)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        map_text = load_map_text(args.map)
    except (OSError, ValueError) as exc:
        print(f"Error reading map file: {exc}")
        return 1

    try:
        result = start_game(args.host, args.port, map_text)
    except requests.RequestException as exc:
        print(f"Failed to start game via API: {exc}")
        return 2

    print("Game started successfully! Response:")
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
