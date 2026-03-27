#!/usr/bin/env python3
"""One-time backfill: generate 400px thumbnails for jobs that predate the
thumbnail_key feature.

For each job that has debug_keys['01_loaded'] but no thumbnail_key:
  1. Download the existing 01_loaded.jpg from S3 (already a JPEG, no HEIC decode needed)
  2. Resize to 400px wide using Pillow
  3. Upload as {user_hash}/thumbnails/{stem}.jpg
  4. Update DynamoDB thumbnail_key

Usage:
    # Dry-run (shows what would happen, no writes)
    python scripts/backfill_thumbnails.py --stage dev --dry-run

    # Live run against dev
    python scripts/backfill_thumbnails.py --stage dev

    # Live run against prod
    python scripts/backfill_thumbnails.py --stage prod
"""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime, timezone

import boto3
from PIL import Image

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIC fallback won't work without pillow-heif, but JPEG sources still will


def _decode_to_jpeg(raw_bytes: bytes, filename: str) -> bytes:
    """Decode any supported format (HEIC, JPEG, PNG) from raw bytes to JPEG bytes."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()

ACCOUNT_ID = "680073251743"
REGION = "us-west-2"


def bucket_name(stage: str) -> str:
    base = f"sundayalbum-data-{ACCOUNT_ID}-{REGION}"
    return base if stage == "prod" else f"{base}-{stage}"
THUMB_WIDTH = 400
THUMB_QUALITY = 85


def table_name(stage: str) -> str:
    return "sa-jobs" if stage == "prod" else f"sa-jobs-{stage}"


def scan_all_jobs(table) -> list[dict]:
    """Scan the full DynamoDB table (handles pagination)."""
    items: list[dict] = []
    kwargs: dict = {}
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        last = resp.get("LastEvaluatedKey")
        if not last:
            break
        kwargs["ExclusiveStartKey"] = last
    return items


def make_thumbnail(jpeg_bytes: bytes) -> bytes:
    """Resize JPEG bytes to THUMB_WIDTH wide, return new JPEG bytes."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    w, h = img.size
    new_h = max(1, int(round(h * THUMB_WIDTH / w)))
    thumb = img.resize((THUMB_WIDTH, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=THUMB_QUALITY, optimize=True)
    return buf.getvalue()


def backfill(stage: str, dry_run: bool) -> None:
    s3 = boto3.client("s3", region_name=REGION)
    dynamo = boto3.resource("dynamodb", region_name=REGION)
    table = dynamo.Table(table_name(stage))
    BUCKET = bucket_name(stage)

    print(f"Scanning {table_name(stage)} in bucket {BUCKET}…")
    items = scan_all_jobs(table)
    print(f"Found {len(items)} total jobs")

    needs_backfill = [
        item for item in items
        if not item.get("thumbnail_key")
        and item.get("debug_keys", {}).get("01_loaded")
    ]
    print(f"{len(needs_backfill)} job(s) need thumbnails")

    if not needs_backfill:
        print("Nothing to do.")
        return

    ok = skipped = errors = 0

    for item in needs_backfill:
        user_hash: str = item["user_hash"]
        job_id: str = item["job_id"]
        stem: str = item.get("input_stem", job_id)
        loaded_rel: str = item["debug_keys"]["01_loaded"]  # e.g. "debug/{stem}_01_loaded.jpg"
        src_key = f"{user_hash}/{loaded_rel}"
        thumb_rel = f"thumbnails/{stem}.jpg"
        thumb_key = f"{user_hash}/{thumb_rel}"  # full key stored in DynamoDB

        print(f"\n  job {job_id[:16]}… stem={stem}")
        print(f"    src  s3://{BUCKET}/{src_key}")
        print(f"    dest s3://{BUCKET}/{thumb_key}")

        if dry_run:
            print("    [DRY RUN] would resize & upload")
            ok += 1
            continue

        # Download existing 01_loaded JPEG; fall back to original upload if expired
        jpeg_bytes: bytes | None = None
        try:
            resp = s3.get_object(Bucket=BUCKET, Key=src_key)
            jpeg_bytes = resp["Body"].read()
            print(f"    using 01_loaded debug JPEG")
        except Exception:
            pass  # will try upload fallback below

        if jpeg_bytes is None:
            # debug image expired — try original upload (30-day lifecycle)
            upload_key = item.get("upload_key", "")
            if upload_key:
                try:
                    resp = s3.get_object(Bucket=BUCKET, Key=upload_key)
                    raw_bytes = resp["Body"].read()
                    print(f"    01_loaded expired, using original upload ({len(raw_bytes)//1024}KB)")
                    # Decode HEIC/JPEG via Pillow (pillow-heif registered below)
                    jpeg_bytes = _decode_to_jpeg(raw_bytes, upload_key)
                except s3.exceptions.NoSuchKey:
                    pass
                except Exception as exc:
                    print(f"    ERROR reading upload fallback: {exc}")

        if jpeg_bytes is None:
            print(f"    SKIP — no source image available (debug expired, upload gone)")
            skipped += 1
            continue

        # Resize to thumbnail
        try:
            thumb_bytes = make_thumbnail(jpeg_bytes)
        except Exception as exc:
            print(f"    ERROR resizing: {exc}")
            errors += 1
            continue

        # Upload thumbnail
        try:
            s3.put_object(
                Bucket=BUCKET,
                Key=thumb_key,
                Body=thumb_bytes,
                ContentType="image/jpeg",
            )
            print(f"    uploaded {len(thumb_bytes) // 1024}KB thumbnail")
        except Exception as exc:
            print(f"    ERROR uploading: {exc}")
            errors += 1
            continue

        # Update DynamoDB
        try:
            now = datetime.now(timezone.utc).isoformat()
            table.update_item(
                Key={"user_hash": user_hash, "job_id": job_id},
                UpdateExpression="SET thumbnail_key = :tk, updated_at = :ua",
                ExpressionAttributeValues={":tk": thumb_key, ":ua": now},
            )
            print(f"    DynamoDB updated")
            ok += 1
        except Exception as exc:
            print(f"    ERROR updating DynamoDB: {exc}")
            errors += 1

    print(f"\nDone. ok={ok}  skipped={skipped}  errors={errors}")
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill job thumbnails")
    parser.add_argument("--stage", choices=["dev", "prod"], default="dev",
                        help="Which environment to backfill (default: dev)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing anything")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no writes will occur ===")
    backfill(args.stage, args.dry_run)


if __name__ == "__main__":
    main()
