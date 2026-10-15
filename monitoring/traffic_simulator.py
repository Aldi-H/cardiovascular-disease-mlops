"""Generate normal, shifted, or invalid synthetic traffic for the API."""

import argparse
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx


def make_payload(scenario: str, index: int, seed: int = 42) -> dict:
    """Create a synthetic request; no patient records are used."""
    rng = random.Random(seed + index)

    if scenario == "drift":
        systolic = rng.randint(180, 250)
        payload = {
            "age_days": rng.randint(22_000, 23_713),
            "gender": rng.choice([1, 2]),
            "height": rng.randint(120, 140),
            "weight": round(rng.uniform(170, 250), 1),
            "ap_hi": systolic,
            "ap_lo": rng.randint(100, min(150, systolic)),
            "cholesterol": 3,
            "gluc": 3,
            "smoke": rng.choice([0, 1]),
            "alco": rng.choice([0, 1]),
            "active": rng.choice([0, 1]),
        }
    else:
        systolic = rng.randint(100, 180)
        payload = {
            "age_days": rng.randint(14_000, 23_000),
            "gender": rng.choice([1, 2]),
            "height": rng.randint(145, 190),
            "weight": round(rng.uniform(45, 120), 1),
            "ap_hi": systolic,
            "ap_lo": rng.randint(60, min(110, systolic)),
            "cholesterol": rng.randint(1, 3),
            "gluc": rng.randint(1, 3),
            "smoke": rng.choice([0, 1]),
            "alco": rng.choice([0, 1]),
            "active": rng.choice([0, 1]),
        }

    if scenario == "invalid":
        # Values are individually in-range, but the systolic/diastolic order is invalid.
        payload["ap_hi"] = 80
        payload["ap_lo"] = 100

    return payload


def send_request(url: str, payload: dict, timeout: float) -> str:
    try:
        response = httpx.post(url, json=payload, timeout=timeout)
        return str(response.status_code)
    except httpx.RequestError:
        return "network_error"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="http://localhost:8000/predict",
        help="FastAPI prediction endpoint.",
    )
    parser.add_argument(
        "--scenario",
        choices=("normal", "drift", "invalid"),
        default="normal",
    )
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)

    if args.requests < 1:
        parser.error("--requests harus minimal 1.")
    if args.concurrency < 1:
        parser.error("--concurrency harus minimal 1.")

    statuses = Counter()

    def submit(index: int) -> str:
        payload = make_payload(args.scenario, index, args.seed)
        return send_request(args.url, payload, args.timeout)

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for status in executor.map(submit, range(args.requests)):
            statuses[status] += 1

    print(f"Scenario: {args.scenario}")
    print(f"Total requests: {args.requests}")
    for status, count in sorted(statuses.items()):
        print(f"HTTP {status}: {count}")


if __name__ == "__main__":
    main()
