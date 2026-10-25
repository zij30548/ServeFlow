"""Run with python -B -m demo_business init|verify; never prints credentials."""
import argparse
import json
import os
import sys

from .initialize import initialize_database, verify_database
from .store import DemoStoreError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Explicitly initialize or verify an isolated ServeFlow MySQL demo database")
    parser.add_argument("operation", choices=("init", "verify"))
    args = parser.parse_args(argv)
    value = os.getenv("SERVEFLOW_DEMO_ADMIN_MYSQL_URL", "") if args.operation == "init" else (
        os.getenv("SERVEFLOW_DEMO_MYSQL_URL", "") or os.getenv("SERVEFLOW_DEMO_ADMIN_MYSQL_URL", "")
    )
    if not value:
        print("STOP: demo_database_url_missing", file=sys.stderr)
        return 2
    try:
        result = initialize_database(value) if args.operation == "init" else verify_database(value)
    except DemoStoreError as exc:
        print(f"STOP: {exc.code}; no automatic retry, reset or database fallback", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
