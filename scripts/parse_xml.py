#!/usr/bin/env python3
"""
Parse JUnit XML test reports into structured JSONL format.

Input:  Directory of JUnit XML files (searched recursively)
Output: data/processed/parsed_failures.jsonl

Each output record:
  suite_name, class_name, test_name, status, duration_sec,
  error_type, error_message, stack_trace, system_out, system_err,
  timestamp, source_file

Usage:
  python scripts/parse_xml.py --xml_dir data/raw --output data/processed/parsed_failures.jsonl
  python scripts/parse_xml.py --xml_dir data/raw --failures_only
"""

import argparse
import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text(elem, tag: str, default: str = "") -> str:
    child = elem.find(tag)
    if child is None:
        return default
    return (child.text or "").strip()


def _attr(elem, key: str, default: str = "") -> str:
    return (elem.get(key) or default).strip()


# ---------------------------------------------------------------------------
# Per-testcase parser
# ---------------------------------------------------------------------------

def parse_testcase(tc, suite_name: str, suite_timestamp: str) -> Optional[dict]:
    """Extract a structured failure record from a <testcase> XML element."""
    test_name  = _attr(tc, "name")
    class_name = _attr(tc, "classname")
    duration   = _attr(tc, "time", "0")

    system_out = _text(tc, "system-out")
    system_err = _text(tc, "system-err")

    failure = tc.find("failure")
    error   = tc.find("error")
    skipped = tc.find("skipped")

    if failure is not None:
        status        = "FAILED"
        error_type    = _attr(failure, "type")
        error_message = _attr(failure, "message")
        stack_trace   = (failure.text or "").strip()
    elif error is not None:
        status        = "ERROR"
        error_type    = _attr(error, "type")
        error_message = _attr(error, "message")
        stack_trace   = (error.text or "").strip()
    elif skipped is not None:
        status        = "SKIPPED"
        error_type    = ""
        error_message = _attr(skipped, "message")
        stack_trace   = ""
    else:
        status        = "PASSED"
        error_type    = ""
        error_message = ""
        stack_trace   = ""

    return {
        "suite_name":    suite_name,
        "class_name":    class_name,
        "test_name":     test_name,
        "status":        status,
        "duration_sec":  duration,
        "error_type":    error_type,
        "error_message": error_message,
        # Preserve up to 200 lines of stack trace; truncate silently
        "stack_trace":   "\n".join(stack_trace.splitlines()[:200]),
        "system_out":    system_out[:3000] if system_out else "",
        "system_err":    system_err[:3000] if system_err else "",
        "timestamp":     suite_timestamp,
    }


# ---------------------------------------------------------------------------
# Per-file parser
# ---------------------------------------------------------------------------

def parse_xml_file(xml_path: Path) -> list[dict]:
    """Parse a single JUnit XML file and return a list of test records."""
    records: list[dict] = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        logger.warning("Skipping malformed XML %s: %s", xml_path, exc)
        return records

    suites: list[ET.Element] = []
    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    elif root.tag == "testsuite":
        suites = [root]
    else:
        logger.warning("Unexpected root tag '%s' in %s; skipping", root.tag, xml_path)
        return records

    for suite in suites:
        suite_name      = _attr(suite, "name", xml_path.stem)
        suite_timestamp = _attr(suite, "timestamp", "")
        for tc in suite.findall("testcase"):
            rec = parse_testcase(tc, suite_name, suite_timestamp)
            if rec:
                rec["source_file"] = str(xml_path.resolve())
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def parse_directory(
    xml_dir: Path,
    output_path: Path,
    failures_only: bool = False,
) -> int:
    """Recursively parse all XML files in xml_dir and write JSONL to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.rglob("*.xml"))
    if not xml_files:
        logger.error("No XML files found under %s", xml_dir)
        sys.exit(1)

    logger.info("Found %d XML files to process", len(xml_files))
    total = written = 0

    with output_path.open("w", encoding="utf-8") as fout:
        for xml_path in xml_files:
            records = parse_xml_file(xml_path)
            for rec in records:
                total += 1
                if failures_only and rec["status"] not in ("FAILED", "ERROR"):
                    continue
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    logger.info(
        "Parsed %d test records total; wrote %d to %s",
        total, written, output_path,
    )
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse JUnit XML test reports into structured JSONL"
    )
    ap.add_argument(
        "--xml_dir", type=Path, required=True,
        help="Root directory containing JUnit XML files (searched recursively)",
    )
    ap.add_argument(
        "--output", type=Path,
        default=Path("data/processed/parsed_failures.jsonl"),
        help="Destination JSONL file (default: data/processed/parsed_failures.jsonl)",
    )
    ap.add_argument(
        "--failures_only", action="store_true", default=False,
        help="Only write FAILED and ERROR cases (skip PASSED / SKIPPED)",
    )
    args = ap.parse_args()
    parse_directory(args.xml_dir, args.output, args.failures_only)


if __name__ == "__main__":
    main()
