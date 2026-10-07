"""Read-only bounded inventory of legacy Redis alert fields; never changes data."""
import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import redis


def inspect_alerts(client, *, data_prefix="alerts:data:", timeline="alerts:timeline",
                   active="alerts:active", limit=500):
    counts = Counter()
    examined = 0
    cursor_exhausted = True
    for key in client.scan_iter(match=data_prefix + "*", count=100):
        if examined >= limit:
            cursor_exhausted = False
            break
        raw = client.hgetall(key)
        examined += 1
        for field in ("acknowledged_at", "resolved_at", "snooze_until", "applied_at",
                      "acknowledged_by", "applied_by"):
            value = raw.get(field)
            if (field.endswith("_at") or field == "snooze_until") and value == "null" or (isinstance(value, str) and re.fullmatch(r"userdata: 0x[0-9a-fA-F]+", value)):
                counts[field] += 1
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=30)
    oldest = client.zrange(timeline, 0, 0, withscores=True)
    return {
        "checked_at_utc": now.isoformat(), "read_only": True,
        "records_examined": examined, "scan_exhausted": cursor_exhausted,
        "sample_limit": limit, "legacy_null_fields_in_sample": dict(counts),
        "timeline_entries": client.zcard(timeline), "active_index_members": client.scard(active),
        "timeline_entries_older_than_30_days": client.zcount(timeline, "-inf", cutoff.timestamp()),
        "oldest_timeline_date_utc": datetime.fromtimestamp(oldest[0][1], timezone.utc).isoformat() if oldest else None,
        "note": "Bounded SCAN sample; index membership does not establish current user-visible alert state",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500, help="Maximum records inspected (1 to 10000)")
    args = parser.parse_args()
    if not 1 <= args.limit <= 10000:
        parser.error("limit must be between 1 and 10000")
    url = os.environ.get("REDIS_URL")
    if not url:
        parser.error("REDIS_URL must identify the intended Redis instance")
    client = redis.from_url(url, decode_responses=True, socket_timeout=3, socket_connect_timeout=3)
    try:
        print(json.dumps(inspect_alerts(client, limit=args.limit), indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
