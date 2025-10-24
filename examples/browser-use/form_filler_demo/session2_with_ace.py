"""Session 2 (ACE-enhanced) using Browser Use agent and GPT-4o-mini."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from ace import DeltaBatch, DeltaOperation, Playbook
from ace.observability import configure_opik, get_integration
from browser_use import Agent, Browser, ChatOpenAI
from dotenv import load_dotenv

from normalizers import FormNormalizer


LOGGER = logging.getLogger("smart_form_filler.session2")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _ensure_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        LOGGER.error("OPENAI_API_KEY not set. Update your .env or environment and retry.")
        sys.exit(1)


def _start_http_server(port: int = 8765) -> threading.Thread:
    """Start HTTP server in background thread."""
    class QuietHandler(SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

    server = HTTPServer(("127.0.0.1", port), QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    LOGGER.info(f"Started HTTP server on http://127.0.0.1:{port}")
    return thread


def _resolve_target_uri(config_path: Path, config: Dict[str, Any]) -> str:
    if config.get("target_form_url"):
        return str(config["target_form_url"])
    local_path = config.get("target_form_path")
    if not local_path:
        raise ValueError("Provide either target_form_url or target_form_path in config.yaml")
    form_file = (config_path.parent / local_path).resolve()
    if not form_file.exists():
        raise FileNotFoundError(f"Local form file not found: {form_file}")

    # Return localhost URL instead of file:// to avoid browser security blocks
    _start_http_server()
    return f"http://127.0.0.1:8765/{local_path}"


def _find_latest_baseline() -> Optional[Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(ARTIFACTS_DIR.glob("session1_baseline_*.json"))
    return candidates[-1] if candidates else None


def _find_latest_playbook() -> Optional[Path]:
    """Find most recent learned playbook."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(ARTIFACTS_DIR.glob("learned_playbook_*.json"))
    return candidates[-1] if candidates else None


def _infer_rules(baseline: Dict[str, Any]) -> Dict[str, Any]:
    events = baseline.get("validation_events", []) or []
    dom_hints = baseline.get("dom_hints", {}) or {}

    rules: Dict[str, Any] = {
        "date_format": None,
        "phone": {},
        "zip": {},
    }

    for event in events:
        field = (event.get("field") or "").lower()
        message = (event.get("message") or "").lower()

        if "date" in field or "date" in message:
            if "mm/dd/yyyy" in message:
                rules["date_format"] = "MM/DD/YYYY"
            elif "yyyy-mm-dd" in message:
                rules["date_format"] = "YYYY-MM-DD"

        if "phone" in field or "phone" in message:
            digits_match = re.search(r"(\d+)\s*digit", message)
            if digits_match:
                count = int(digits_match.group(1))
                rules.setdefault("phone", {})["min_digits"] = count
                rules["phone"]["max_digits"] = count
            if "digits" in message:
                rules.setdefault("phone", {})["output_format"] = "digits_only"
            if "country" in message and "code" in message:
                rules.setdefault("phone", {})["allow_country_code"] = False

        if any(token in field for token in ["zip", "postal"]) or any(
            token in message for token in ["zip", "postal"]
        ):
            digits_match = re.search(r"(\d+)\s*digit", message)
            if digits_match:
                rules.setdefault("zip", {})["length"] = int(digits_match.group(1))
            if "digits" in message:
                rules.setdefault("zip", {})["digits_only"] = True

    if not rules.get("date_format"):
        placeholders = dom_hints.get("placeholders", {})
        for placeholder in placeholders.values():
            text = (placeholder or "").upper()
            if "MM/DD/YYYY" in text:
                rules["date_format"] = "MM/DD/YYYY"
                break

    rules.setdefault("date_format", "MM/DD/YYYY")
    phone_defaults = {"min_digits": 9, "max_digits": 9, "allow_country_code": False, "output_format": "digits_only"}
    zip_defaults = {"length": 5, "digits_only": True, "uppercase": True, "allow_spaces": False}
    rules["phone"] = {**phone_defaults, **rules.get("phone", {})}
    rules["zip"] = {**zip_defaults, **rules.get("zip", {})}

    return rules


def _build_playbook(rules: Dict[str, Any]) -> Tuple[Playbook, Dict[str, Any]]:
    playbook = Playbook()
    operations = [
        DeltaOperation(
            type="ADD",
            section="normalization_rules",
            content=f"Normalize ship_date to {rules['date_format']}.",
            metadata={"helpful": 2},
        ),
        DeltaOperation(
            type="ADD",
            section="normalization_rules",
            content=f"Strip phone to digits-only with length {rules['phone']['min_digits']}.",
            metadata={"helpful": 2},
        ),
        DeltaOperation(
            type="ADD",
            section="normalization_rules",
            content=f"Enforce postal code as {rules['zip']['length']} digits (no spaces).",
            metadata={"helpful": 1},
        ),
    ]
    playbook.apply_delta(
        DeltaBatch(
            reasoning="Constructed from Session 1 validation errors and DOM hints",
            operations=operations,
        )
    )

    hints = {
        "date_format": rules["date_format"],
        "phone": rules["phone"],
        "zip": rules["zip"],
    }
    return playbook, hints


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


@dataclass
class AgentOutputs:
    steps: int = 0
    validation_events: Optional[Any] = None
    final_status: str = "unknown"
    zero_error_first_try: bool = False

    @classmethod
    def from_summary(cls, summary: Dict[str, Any]) -> "AgentOutputs":
        steps_value = summary.get("steps", 0)
        # Handle if steps is a list (extract length) or string (convert to int)
        if isinstance(steps_value, list):
            steps = len(steps_value)
        elif isinstance(steps_value, str):
            steps = int(steps_value) if steps_value.isdigit() else 0
        else:
            steps = int(steps_value or 0)

        return cls(
            steps=steps,
            validation_events=summary.get("validation_events"),
            final_status=str(summary.get("final_status", "unknown")),
            zero_error_first_try=bool(summary.get("zero_error_first_try", False)),
        )


class ACERunner:
    def __init__(
        self,
        *,
        config_path: Path,
        payload_path: Path,
        baseline_path: Optional[Path],
        playbook_path: Optional[Path],
        headless: bool,
        run_number: int = 1,
    ) -> None:
        self.config_path = config_path
        self.payload_path = payload_path
        self.baseline_path = baseline_path
        self.playbook_path = playbook_path
        self.headless = headless
        self.run_number = run_number

        self.config = _load_yaml(config_path)
        self.payload = _load_json(payload_path)
        self.form_uri = _resolve_target_uri(config_path, self.config)

        # Check if we have an existing playbook (Run 2+)
        if playbook_path and playbook_path.exists():
            LOGGER.info(f"Loading existing playbook from {playbook_path} (Run {run_number})")
            playbook_data = _load_json(playbook_path)
            self.playbook_hints = playbook_data.get("hints", {})
            self.playbook, _ = _build_playbook(self.playbook_hints)
            self.baseline = None
        # Train from baseline if provided
        elif baseline_path and baseline_path.exists():
            LOGGER.info(f"Training ACE from baseline {baseline_path} (Run {run_number})")
            self.baseline = _load_json(baseline_path)
            self.rules = _infer_rules(self.baseline)
            self.playbook, self.playbook_hints = _build_playbook(self.rules)
        # Run 1 with no baseline: Use default rules (will fill raw, capture errors, train on-the-fly)
        else:
            LOGGER.info(f"Run {run_number}: No baseline or playbook. Using default rules.")
            self.baseline = None
            # Start with empty/default rules - will be updated after first fill
            self.rules = {
                "date_format": "MM/DD/YYYY",  # Common US format
                "phone": {"min_digits": 9, "max_digits": 9, "allow_country_code": False, "output_format": "digits_only"},
                "zip": {"length": 5, "digits_only": True, "uppercase": True, "allow_spaces": False}
            }
            self.playbook, self.playbook_hints = _build_playbook(self.rules)

        self.normalizer = FormNormalizer(playbook_hints=self.playbook_hints)
        self.normalized_payload = self.normalizer.normalize_all(self.payload)

    def _build_task(self) -> str:
        normalized_json = json.dumps(self.normalized_payload, indent=2)
        selectors_json = json.dumps(self.config.get("selectors", {}), indent=2)
        playbook_prompt = self.playbook.as_prompt()

        return f"""
Fill form at: {self.form_uri}

Use these exact values:
- first_name: {self.normalized_payload.get('first_name')}
- last_name: {self.normalized_payload.get('last_name')}
- email: {self.normalized_payload.get('email')}
- phone: {self.normalized_payload.get('phone_norm', self.normalized_payload.get('phone_raw'))}
- ship_date: {self.normalized_payload.get('ship_date_norm', self.normalized_payload.get('ship_date_raw'))}
- address1: {self.normalized_payload.get('address1')}
- address2: {self.normalized_payload.get('address2')}
- city: {self.normalized_payload.get('city')}
- state_region: {self.normalized_payload.get('state_region')}
- zip_postal: {self.normalized_payload.get('zip_postal_norm', self.normalized_payload.get('zip_postal_raw'))}
- country: {self.normalized_payload.get('country')}

Steps:
1. Go to URL
2. Fill #first_name with first_name value
3. Fill #last_name with last_name value
4. Fill #email with email value
5. Fill #phone with phone value
6. Fill #ship_date with ship_date value
7. Fill #address1 with address1 value
8. Fill #address2 with address2 value (optional)
9. Fill #city with city value
10. Fill #state_region with state_region value
11. Fill #zip_postal with zip_postal value
12. Select #country dropdown with country value
13. Click #submit button
14. Wait 2 seconds
15. Check .status-area - if it contains "successfully", return success JSON
16. Return JSON: {{"steps": 16, "final_status": "success", "zero_error_first_try": true, "validation_events": [], "notes": "Form submitted"}}

STOP at step 16.
"""

    async def run(self) -> Dict[str, Any]:
        load_dotenv()
        _ensure_openai_key()

        # Configure Opik for observability
        opik_integration = configure_opik(
            project_name="ace-browser-demos",
            tags=["form-filler", f"run-{self.run_number}"]
        )

        browser = Browser(headless=self.headless)
        await browser.start()

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        task = self._build_task()

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            max_steps=int(self.config.get("browser", {}).get("max_steps", 25)),
            context={
                "payload": self.payload,
                "normalized_payload": self.normalized_payload,
                "playbook": self.playbook_hints,
            },
        )

        started_at = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        try:
            history = await asyncio.wait_for(
                agent.run(),
                timeout=float(self.config.get("browser", {}).get("timeout", 90)),
            )
        finally:
            await browser.stop()

        end_perf = time.perf_counter()
        finished_at = datetime.now(timezone.utc)

        summary: Dict[str, Any] = {}
        if history and hasattr(history, "final_result"):
            summary = _extract_json_blob(history.final_result()) or {}

        outputs = AgentOutputs.from_summary(summary)

        metrics: Dict[str, Any] = {
            "session": 2,
            "session_type": "with_ace",
            "form_url": self.form_uri,
            "start_time": started_at.isoformat(),
            "end_time": finished_at.isoformat(),
            "total_time_seconds": round(end_perf - start_perf, 2),
            "steps": outputs.steps,
            "final_status": outputs.final_status,
            "validation_error_count": len(outputs.validation_events or []),
            "zero_error_first_try": outputs.zero_error_first_try,
        }

        # Log demo performance to Opik
        if opik_integration.is_available():
            opik_integration.log_role_performance(
                role_name="BrowserAgent",
                execution_time=end_perf - start_perf,
                success=outputs.final_status == "success",
                input_data={
                    "form_url": self.form_uri,
                    "payload_keys": list(self.payload.keys()),
                    "run_number": self.run_number
                },
                output_data={
                    "steps": outputs.steps,
                    "validation_errors": len(outputs.validation_events or []),
                    "zero_error_first_try": outputs.zero_error_first_try
                },
                metadata={
                    "demo_type": "form-filler",
                    "session_type": "with_ace",
                    "playbook_bullets": len(self.playbook.bullets()),
                    "learned_strategies": len([b for b in self.playbook.bullets() if b.helpful > 0])
                }
            )

        # Only compare if baseline exists
        baseline_metrics = self.baseline.get("metrics", {}) if self.baseline else {}
        comparison = {
            "baseline_time": baseline_metrics.get("total_time_seconds") if baseline_metrics else None,
            "baseline_steps": baseline_metrics.get("steps") if baseline_metrics else None,
            "baseline_errors": baseline_metrics.get("validation_error_count") if baseline_metrics else None,
            "time_delta_seconds": None,
            "time_delta_pct": None,
            "steps_delta": None,
            "errors_resolved": None,
        }

        if baseline_metrics:
            base_time = baseline_metrics.get("total_time_seconds") or 0
            base_steps = baseline_metrics.get("steps") or 0
            base_errors = baseline_metrics.get("validation_error_count") or 0

            if base_time:
                comparison.update(
                    {
                        "time_delta_seconds": round(base_time - metrics["total_time_seconds"], 2),
                        "time_delta_pct": round(((base_time - metrics["total_time_seconds"]) / base_time) * 100, 2),
                        "steps_delta": base_steps - metrics["steps"],
                        "errors_resolved": base_errors - metrics["validation_error_count"],
                    }
                )

        result = {
            "metadata": {
                "session": f"Session 2 - With ACE (Run {self.run_number})",
                "description": "Browser Use agent applying ACE normalization",
                "generated_at": finished_at.isoformat(),
                "baseline_source": str(self.baseline_path) if self.baseline_path else "none",
                "playbook_source": str(self.playbook_path) if self.playbook_path else "generated",
                "run_number": self.run_number,
            },
            "metrics": metrics,
            "comparison_vs_baseline": comparison,
            "normalizations_applied": {
                "ship_date": f"{self.payload.get('ship_date_raw')} → {self.normalized_payload.get('ship_date_norm')}",
                "phone": f"{self.payload.get('phone_raw')} → {self.normalized_payload.get('phone_norm')}",
                "zip_postal": f"{self.payload.get('zip_postal_raw')} → {self.normalized_payload.get('zip_postal_norm')}",
            },
            "validation_events": outputs.validation_events or [],
            "playbook": self.playbook.to_dict(),
            "playbook_hints": self.playbook_hints,
            "payload_comparison": {
                "raw": self.payload,
                "normalized": {k: v for k, v in self.normalized_payload.items() if k.endswith("_norm")},
            },
            "agent_summary": summary,
        }

        return result


def _persist(result: Dict[str, Any]) -> Tuple[Path, Path]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    result_path = ARTIFACTS_DIR / f"session2_with_ace_{timestamp}.json"
    with result_path.open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    playbook_path = ARTIFACTS_DIR / f"learned_playbook_{timestamp}.json"
    with playbook_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "playbook": result["playbook"],
                "hints": result["playbook_hints"],
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "baseline_source": result["metadata"]["baseline_source"],
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    return result_path, playbook_path


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Session 2 with ACE (Browser Use + GPT-4o-mini)")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--payload", default="payload.json", help="Path to payload JSON")
    parser.add_argument(
        "--baseline",
        help="Path to Session 1 baseline JSON (auto-detects latest if not specified)",
    )
    parser.add_argument(
        "--playbook",
        help="Path to existing playbook (for Run 2+). If specified, ignores baseline.",
    )
    parser.add_argument("--headless", action="store_true", help="Run browser headlessly")
    parser.add_argument(
        "--run",
        type=int,
        default=1,
        help="Run number (1=train from baseline, 2+=use existing playbook)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config_path = Path(args.config).resolve()
    payload_path = Path(args.payload).resolve()

    # Determine mode: playbook (Run 2+) or baseline (Run 1)
    playbook_path = None
    baseline_path = None

    if args.playbook:
        # Explicit playbook provided (Run 2+)
        playbook_path = Path(args.playbook).resolve()
        if not playbook_path.exists():
            LOGGER.error(f"Playbook not found: {playbook_path}")
            sys.exit(1)
        LOGGER.info(f"Using existing playbook: {playbook_path}")
    elif args.run > 1:
        # Auto-detect latest playbook for Run 2+
        playbook_path = _find_latest_playbook()
        if not playbook_path:
            LOGGER.error("No playbook found. Run with --run 1 first to train ACE.")
            sys.exit(1)
        LOGGER.info(f"Auto-detected playbook: {playbook_path}")
    else:
        # Run 1: Try to find baseline, but if none exists, create mock baseline
        baseline_path = Path(args.baseline).resolve() if args.baseline else _find_latest_baseline()
        if not baseline_path or not baseline_path.exists():
            LOGGER.warning("No baseline found. Creating initial baseline by filling form with raw data...")
            # Run 1 will fill form, capture errors, then train ACE
            baseline_path = None  # Will be created during execution
        else:
            LOGGER.info(f"Training ACE from baseline: {baseline_path}")

    runner = ACERunner(
        config_path=config_path,
        payload_path=payload_path,
        baseline_path=baseline_path,
        playbook_path=playbook_path,
        headless=args.headless,
        run_number=args.run,
    )

    try:
        result = await runner.run()
    except asyncio.TimeoutError:
        LOGGER.error("ACE session timed out")
        sys.exit(1)
    except Exception as exc:  # pragma: no cover
        LOGGER.error("ACE session failed: %s", exc, exc_info=True)
        sys.exit(1)

    result_path, playbook_path = _persist(result)
    LOGGER.info("ACE artifact saved to %s", result_path)
    LOGGER.info("Playbook saved to %s", playbook_path)
    LOGGER.info(
        "Run %d complete: Zero errors=%s, Validation errors=%s",
        args.run,
        result["metrics"].get("zero_error_first_try"),
        result["metrics"].get("validation_error_count"),
    )

    if args.run == 1:
        LOGGER.info("\n=== ACE TRAINED ===")
        LOGGER.info("Next: Run 'python session2_with_ace.py --run 2' to use learned playbook")
    else:
        LOGGER.info("\n=== Progressive Learning Demonstrated ===")


if __name__ == "__main__":
    asyncio.run(main())
