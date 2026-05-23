"""Simulator that posts tickets to the FastAPI server and prints results."""
import json
import time
import argparse
import httpx
from rich import print
from pathlib import Path

DEFAULT_URL = "http://localhost:8000/resolve"


def main(server_url: str, mode: str, delay: float):
    tickets_file = Path("CAG") / "tickets.json"
    if not tickets_file.exists():
        tickets_file = Path("RAG") / "tickets.json"
    data = json.loads(tickets_file.read_text(encoding="utf-8"))
    tickets = data.get("tickets", [])

    client = httpx.Client(timeout=30.0)
    for ticket in tickets:
        print(f"[cyan]Posting {ticket['ticket_id']} -> mode={mode}[/cyan]")
        try:
            resp = client.post(server_url, json=ticket, params={"mode": mode})
            if resp.status_code == 200:
                try:
                    text = resp.text
                except Exception:
                    text = resp.content.decode("utf-8", errors="ignore")
                print(f"[green]{ticket['ticket_id']} response:[/green]\n{text[:400]}\n")
            else:
                print(f"[red]Error {resp.status_code}: {resp.text}[/red]")
        except Exception as e:
            print(f"[red]Request failed: {e}[/red]")
        time.sleep(delay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=DEFAULT_URL)
    parser.add_argument("--mode", choices=("cag", "rag"), default="cag")
    parser.add_argument("--delay", type=float, default=2.0)
    args = parser.parse_args()
    main(args.server, args.mode, args.delay)
