"""
Submit your solution to the Cactus Evals leaderboard.

Usage:
    python submit.py --team "YourTeamName" --location "SF"
"""

import argparse
import os
import time
import requests

SERVER_URL = "https://cactusevals.ngrok.app"
HEADERS = {"ngrok-skip-browser-warning": "true"}

# Path to main.py: same directory as this script (works from any cwd)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY_PATH = os.path.join(_SCRIPT_DIR, "main.py")


def submit(team, location):
    print("=" * 60)
    print(f"  Submitting main.py for team '{team}' ({location})")
    print("=" * 60)

    try:
        with open(MAIN_PY_PATH, "rb") as f:
            resp = requests.post(
                f"{SERVER_URL}/eval/submit",
                data={"team": team, "location": location},
                files={"file": ("main.py", f, "text/x-python")},
                headers=HEADERS,
                timeout=15,
            )
    except requests.exceptions.ConnectionError:
        print("The Leaderboard is not accepting submissions at this time.")
        return
    except requests.exceptions.Timeout:
        print("The Leaderboard is not accepting submissions at this time.")
        return

    if resp.status_code != 200:
        try:
            msg = resp.json().get("error", resp.text)
        except (requests.exceptions.JSONDecodeError, ValueError):
            print("The Leaderboard is not accepting submissions at this time.")
            return
        print(f"Error: {msg}")
        return

    data = resp.json()
    submission_id = data["submission_id"]
    print(f"Queued! Position: #{data['position_in_queue']}")
    print(f"Submission ID: {submission_id}")
    print(f"\nWaiting for evaluation to complete...\n")

    last_progress = ""
    poll_interval = 3
    max_status_failures = 20
    status_failures = 0
    while True:
        time.sleep(poll_interval)
        try:
            resp = requests.get(
                f"{SERVER_URL}/eval/status",
                params={"id": submission_id},
                headers=HEADERS,
                timeout=15,
            )
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            status_failures += 1
            if status_failures >= max_status_failures:
                print("\nConnection issues while waiting for results. Your submission was queued successfully.")
                print(f"  Submission ID: {submission_id}")
                print("  Check the leaderboard later for your score.")
                return
            print(f"  Network/SSL issue ({status_failures}/{max_status_failures}), retrying in {poll_interval}s...")
            continue
        status_failures = 0

        if resp.status_code != 200:
            print("Error polling status. Retrying...")
            continue

        try:
            status = resp.json()
        except ValueError:
            continue

        if status.get("progress") and status["progress"] != last_progress:
            last_progress = status["progress"]
            print(f"  [{status['progress']}]", flush=True)

        if status.get("status") == "complete":
            result = status.get("result", {})
            print(f"\n{'=' * 50}")
            print(f"  RESULTS for team '{result.get('team', team)}'")
            print(f"{'=' * 50}")
            print(f"  Total Score : {result.get('score', 0):.1f}%")
            print(f"  Avg F1      : {result.get('f1', 0):.4f}")
            print(f"  Avg Time    : {result.get('avg_time_ms', 0):.0f}ms")
            print(f"  On-Device   : {result.get('on_device_pct', 0):.0f}%")
            print(f"  Leaderboard : Updated!")
            print(f"{'=' * 50}")
            return

        if status.get("status") == "error":
            print(f"\nError: {status.get('error', 'Unknown error')}")
            return

        if status.get("status") == "queued":
            print(f"  Queued (queue size: {status.get('queue_size', '?')})...", end="\r", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit to Cactus Evals Leaderboard")
    parser.add_argument("--team", type=str, required=True, help="Your team name")
    parser.add_argument("--location", type=str, required=True, help="Your location (e.g. SF, NYC, London)")
    args = parser.parse_args()
    submit(args.team, args.location)
