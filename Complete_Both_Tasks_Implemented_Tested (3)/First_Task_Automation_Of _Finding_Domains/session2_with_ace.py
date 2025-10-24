"""
Session 2 (WITH ACE): Domain Availability Checker WITH ACE Learning
Uses Generator→Reflector→Curator to learn and optimize over time.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from browser_use import Agent, Browser, ChatOpenAI
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.table import Table

from ace import LiteLLMClient, Generator, Reflector, Curator, Playbook

# ----------------------------
# Constants / configuration
# ----------------------------

# No hardcoded site preferences - let ACE discover and learn optimal sites

MAX_STEPS_TARGET = 8
INEFF_THRESHOLD = 10     # trigger learning if steps > this

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
load_dotenv()


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class DomainCheckResult:
    domain: str
    status: str = "ERROR"
    steps: int = 0

    # Domain checking time tracking
    domain_check_time_seconds: float = 0.0  # Pure domain checking time
    total_time_seconds: float = 0.0         # Combined time

    # Legacy field for backwards compatibility
    time_seconds: float = 0.0  # Same as total_time_seconds

    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    error_message: Optional[str] = None
    site_used: Optional[str] = None
    tokens_used: int = 0
    attempts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ----------------------------
# Persistence for ACE
# ----------------------------

class ACEPersistence:
    def __init__(self, state_dir: Path = Path("ace_state")) -> None:
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.playbook_json = self.state_dir / "learned_playbook.json"
        self.site_performance = self.state_dir / "site_performance.json"

    def save_playbook(self, playbook: Playbook) -> None:
        bullets = [
            {"section": b.section, "content": b.content, "metadata": getattr(b, "metadata", None)}
            for b in playbook.bullets()
        ]
        self.playbook_json.write_text(json.dumps({"bullets": bullets}, indent=2), encoding="utf-8")

    def load_playbook(self, playbook: Playbook) -> bool:
        try:
            if not self.playbook_json.exists():
                return False
            data = json.loads(self.playbook_json.read_text(encoding="utf-8"))
            for b in data.get("bullets", []):
                playbook.add_bullet(b.get("section", "General"), b.get("content", ""), b.get("metadata"))
            logger.info(f"[green]✓ Loaded {len(data.get('bullets', []))} learned strategies[/green]", extra={"markup": True})
            return True
        except Exception as e:
            logger.debug(f"Failed to load playbook: {e}")
            return False

    def update_site_performance(self, site: str, success: bool, steps: int, time_seconds: float) -> None:
        try:
            perf: Dict[str, Any] = {}
            if self.site_performance.exists():
                perf = json.loads(self.site_performance.read_text(encoding="utf-8"))

            if site not in perf:
                perf[site] = {"attempts": 0, "successes": 0, "total_steps": 0, "total_time": 0.0}

            perf[site]["attempts"] += 1
            if success:
                perf[site]["successes"] += 1
            perf[site]["total_steps"] += steps
            perf[site]["total_time"] += time_seconds

            self.site_performance.write_text(json.dumps(perf, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to update site performance: {e}")

    def get_best_sites(self) -> List[str]:
        """Get sites ranked by success rate, then by efficiency (fewer steps)."""
        try:
            if self.site_performance.exists():
                perf = json.loads(self.site_performance.read_text(encoding="utf-8"))
                ranked = sorted(
                    perf.items(),
                    key=lambda x: (
                        x[1]["successes"] / max(1, x[1]["attempts"]),
                        -x[1]["total_steps"] / max(1, x[1]["attempts"]),
                    ),
                    reverse=True,
                )
                return [site for site, stats in ranked if stats["successes"] > 0]
        except Exception as e:
            logger.debug(f"Failed to load site performance: {e}")
        return []


# ----------------------------
# Main checker with ACE
# ----------------------------

class DomainCheckerWithACE:
    """
    Domain checker WITH ACE learning.
    Learns from each attempt to improve future performance.
    """

    def __init__(
        self,
        domains_file: str = "domains.txt",
        model: str = "gpt-4o",
        ace_model: str = "gpt-4o-mini",
        headless: bool = False,
        max_retries: int = 2,
    ) -> None:
        self.domains_file = Path(domains_file)
        self.model = model
        self.ace_model = ace_model
        self.headless = headless
        self.max_retries = max_retries

        self.domains: List[str] = []
        self.results: List[DomainCheckResult] = []
        self._learning_tasks: List[asyncio.Task] = []  # <- collect background learning tasks

        self.metrics: Dict[str, Any] = {
            "session": 2,
            "session_type": "with_ace_learning",
            "start_time": None,
            "end_time": None,
            "total_time_seconds": 0.0,
            "domains_checked": 0,
            "domains_available": 0,
            "domains_taken": 0,
            "errors": 0,
            "total_steps": 0,
            "total_tokens": 0,
            "accuracy_rate": 0.0,
            "average_time_per_domain": 0.0,
            "average_domain_check_time": 0.0,    # New: avg pure domain checking time
            "total_domain_check_time": 0.0,      # New: total domain checking time
            "average_steps_per_domain": 0.0,
            "average_tokens_per_domain": 0.0,
            "ace_improvements_applied": 0,    # counts successful checks (signal of ACE guidance applied)
            "learning_updates_applied": 0,    # counts deltas actually added to playbook this run
            "model": model,
            "ace_model": ace_model,
        }

        # ACE components
        logger.info("[bold cyan]Initializing ACE Framework...[/bold cyan]", extra={"markup": True})
        self.ace_client = LiteLLMClient(model=ace_model)
        self.generator = Generator(self.ace_client)
        self.reflector = Reflector(self.ace_client)
        self.curator = Curator(self.ace_client)
        self.playbook = Playbook()

        # Persistence
        self.persistence = ACEPersistence()
        self.persistence.load_playbook(self.playbook)

        # Load session 1 baseline for context
        self._load_baseline_context()

        logger.info(f"[green]✓ ACE Domain Checker Initialized[/green]", extra={"markup": True})
        logger.info(f"Model: {model} | ACE Model: {ace_model} | Learning: Enabled")

    def _load_baseline_context(self) -> None:
        """Load Session 1 results to warm-start ACE learning, with robust fallbacks."""
        try:
            baseline_files = list(Path(".").glob("session1_baseline_*.json"))
            if not baseline_files:
                return

            latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)
            with open(latest_baseline, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)

            metrics = baseline_data.get("metrics", {}) or {}

            # If Session 1 forgot to write average_steps_per_domain, derive from results
            baseline_steps = metrics.get("average_steps_per_domain")
            if baseline_steps is None:
                results = baseline_data.get("results", []) or []
                if results:
                    steps_sum = sum(int(r.get("steps", 0)) for r in results)
                    baseline_steps = steps_sum / max(1, len(results))
                    metrics["average_steps_per_domain"] = baseline_steps
                else:
                    baseline_steps = 0.0

            logger.info(
                f"[cyan]Loaded Session 1 baseline: {baseline_steps:.1f} avg steps[/cyan]",
                extra={"markup": True}
            )

            # Add baseline context to playbook only once
            if len(self.playbook.bullets()) == 0:
                self.playbook.add_bullet(
                    section="Baseline Context",
                    content=(
                        f"Session 1 baseline: {baseline_steps:.1f} avg steps, "
                        f"{metrics.get('average_time_per_domain', 0):.1f}s avg time. "
                        f"Success rate: {metrics.get('accuracy_rate', 0)}%. "
                        f"Goal: Improve these metrics."
                    ),
                )
        except Exception as e:
            logger.debug(f"Could not load baseline context: {e}")

    def load_domains(self) -> List[str]:
        if not self.domains_file.exists():
            raise FileNotFoundError(f"Domains file not found: {self.domains_file}")

        with self.domains_file.open("r", encoding="utf-8") as f:
            domains = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not domains:
            raise ValueError(f"No valid domains found in {self.domains_file}")

        self.domains = domains
        logger.info(f"✓ Loaded {len(domains)} domains")

        table = Table(title="Domains to Check", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Domain", style="cyan")
        for idx, d in enumerate(domains, 1):
            table.add_row(str(idx), d)
        console.print(table)

        return domains

    async def _generate_optimized_strategy(self, domain: str) -> str:
        """Use Generator to create optimized strategy based on learned playbook."""
        try:
            best_sites = self.persistence.get_best_sites()

            context = "Find and use reliable domain lookup websites. "
            if best_sites:
                context += f"Previously successful sites (learned from experience): {', '.join(best_sites[:3])}. "
            else:
                context += "Discover which domain lookup sites work best through trial and error. "

            context += "Goal: Check domain efficiently with minimum steps. Learn which sites to use and which to avoid."

            generated = await asyncio.to_thread(
                self.generator.generate,
                question=f"How to efficiently check if domain {domain} is available?",
                context=context,
                playbook=self.playbook,
            )

            return generated
        except Exception as e:
            logger.debug(f"Strategy generation failed: {e}")
            return "Find a reliable domain lookup website and check domain availability efficiently."

    def _extract_step_count(self, history: Any) -> int:
        """Robust step counting across possible history APIs."""
        try:
            if hasattr(history, "action_names"):
                names = history.action_names()
                return len(names) if names is not None else 0
            # fallback: try len(history) if it is a sequence-like object
            return len(history) if history is not None else 0
        except Exception:
            return 0

    def _detect_site_used(self, history: Any) -> Optional[str]:
        """Scan visited URLs to identify which site was used for the domain check."""
        try:
            urls = history.urls() if hasattr(history, "urls") else None
            if not urls:
                return None

            # Common domain lookup sites
            known_sites = [
                "who.is", "whois.com", "domainr.com", "instantdomainsearch.com",
                "whois.net", "godaddy.com", "namecheap.com", "domain.com",
                "networksolutions.com", "freenom.com"
            ]

            for url in reversed(urls):
                u = url.lower()
                for site in known_sites:
                    if site in u:
                        return site

            # If no known site found, extract domain from last meaningful URL
            for url in reversed(urls):
                if "://" in url and not url.startswith("data:"):
                    try:
                        from urllib.parse import urlparse
                        parsed = urlparse(url)
                        if parsed.netloc:
                            return parsed.netloc.lower()
                    except:
                        pass
            return None
        except Exception:
            return None

    async def check_single_domain(
        self,
        domain: str
    ) -> DomainCheckResult:
        """
        Check domain WITH ACE - learns from each attempt, optimizes over time.
        """
        overall_start_time = time.time()
        domain_check_start = time.time()
        attempt = 1
        final_result = DomainCheckResult(domain=domain)
        total_steps = 0
        total_tokens = 0
        learning_tasks_for_this_domain = []

        while attempt <= self.max_retries + 1:
            attempt_result = await self._check_single_attempt(domain, attempt, learning_tasks_for_this_domain)

            # Accumulate steps and tokens across all attempts
            total_steps += attempt_result.steps
            total_tokens += attempt_result.tokens_used

            # Always update the final result with the latest attempt data
            final_result.status = attempt_result.status
            final_result.steps = total_steps  # Total steps across all attempts
            final_result.error_message = attempt_result.error_message
            final_result.site_used = attempt_result.site_used
            final_result.tokens_used = total_tokens  # Total tokens across all attempts
            final_result.attempts = attempt

            # If successful, break out
            if attempt_result.status != 'ERROR':
                break

            # If this was the last attempt, break out
            if attempt > self.max_retries:
                break

            # Prepare for retry
            logger.info(f"Retrying {domain}... (Attempt {attempt + 1}/{self.max_retries + 1})")
            await asyncio.sleep(2)
            attempt += 1

        # Calculate domain checking time (excludes ACE learning)
        domain_check_end = time.time()
        final_result.domain_check_time_seconds = round(domain_check_end - domain_check_start, 2)

        # Wait for ACE learning tasks
        if learning_tasks_for_this_domain:
            await asyncio.gather(*learning_tasks_for_this_domain, return_exceptions=True)

        # Calculate total time
        final_result.total_time_seconds = round(time.time() - overall_start_time, 2)
        final_result.time_seconds = final_result.total_time_seconds  # Backwards compatibility

        return final_result

    async def _check_single_attempt(
        self,
        domain: str,
        attempt: int = 1,
        learning_tasks_for_domain: Optional[List] = None
    ) -> DomainCheckResult:
        """
        Single attempt at checking a domain (used by check_single_domain).
        """
        result = DomainCheckResult(domain=domain, attempts=attempt)
        browser: Optional[Browser] = None

        logger.info(f"\n{'='*60}")
        logger.info(f"Checking: {domain} (Attempt {attempt}/{self.max_retries + 1})")
        logger.info(f"{'='*60}")

        # Generate ACE-optimized strategy
        logger.info("[cyan]Generating ACE-optimized strategy...[/cyan]", extra={"markup": True})
        strategy = await self._generate_optimized_strategy(domain)

        try:
            browser = Browser(headless=self.headless)
            await browser.start()

            llm = ChatOpenAI(model=self.model, temperature=0.0)


            # ACE-ENHANCED TASK: Includes learned strategies
            playbook_guidance = ""
            if len(self.playbook.bullets()) > 0:
                playbook_guidance = "\n\nLEARNED STRATEGIES:\n"
                for bullet in self.playbook.bullets()[:5]:  # Top 5 strategies
                    playbook_guidance += f"• {bullet.content}\n"

            # Get recommended sites from ACE learning
            best_sites = self.persistence.get_best_sites()
            learned_guidance = ""
            if best_sites:
                learned_guidance = f"\n\nLEARNED SUCCESSFUL SITES (from ACE experience):\n"
                for i, site in enumerate(best_sites[:3], 1):
                    learned_guidance += f"{i}. {site} (proven to work)\n"

            # Base task prompt (consistent with Session 1)
            base_task = f"""
You are a domain availability checker agent. Determine if "{domain}" is available.

TASK: Find a reliable domain lookup website and check if the domain is available.

INSTRUCTIONS:
1) Navigate to a domain availability checker website
3) Find the domain search functionality on the chosen site
4) Enter exactly: {domain}
5) Submit the search and wait for the results to fully load
6) Determine the status from clear indicators such as: "Available", "Add to Cart", "Taken", "Unavailable", "Premium", "Already Registered"
7) If results are unclear, try a different lookup site
8) Double-check the result corresponds to the exact domain

GOAL: Be efficient and find a working site that gives clear results.

OUTPUT FORMAT (exactly one line at the end):
- "AVAILABLE: {domain}"
- "TAKEN: {domain}"
- "ERROR: Could not determine status for {domain}"
"""

            # ACE enhancements (appended to base task)
            ace_enhancements = f"""

ACE-OPTIMIZED STRATEGY:
{strategy}

{learned_guidance}

{playbook_guidance}

EFFICIENCY TARGET: Complete in 5-8 steps."""

            task = base_task + ace_enhancements

            # ACE-optimized agent configuration
            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_actions_per_step=5,  # Focused actions
                max_steps=30,            # Guardrail
            )

            t0 = time.time()
            history = await asyncio.wait_for(agent.run(), timeout=180.0)  # 3 minutes
            exec_time = round(time.time() - t0, 2)
            step_count = self._extract_step_count(history)

            # Get token usage from history.usage - this is the correct way for browser_use
            tokens_used = 0
            if hasattr(history, 'usage') and history.usage:
                tokens_used = history.usage.total_tokens

            # Fallback: Try to get token usage from agent if available
            if tokens_used == 0 and hasattr(agent, 'token_cost_service'):
                # Try to get from the token cost service
                try:
                    cost_service = getattr(agent, 'token_cost_service', None)
                    if cost_service and hasattr(cost_service, 'get_total_usage'):
                        usage = cost_service.get_total_usage()
                        tokens_used = getattr(usage, 'total_tokens', 0)
                except Exception:
                    pass

            result.steps = step_count
            result.time_seconds = exec_time
            result.tokens_used = tokens_used

            # Identify which site was used (robust across redirects)
            result.site_used = self._detect_site_used(history)

            # Parse output
            final_text = history.final_result() if hasattr(history, "final_result") else ""
            status = self._parse_agent_output(final_text or "", domain)
            result.status = status

            # Update site performance tracking
            if result.site_used:
                self.persistence.update_site_performance(
                    result.site_used,
                    success=(status in ["AVAILABLE", "TAKEN"]),
                    steps=step_count,
                    time_seconds=exec_time,
                )

            # ACE LEARNING: Reflect and improve (schedule for this domain)
            if learning_tasks_for_domain is not None:
                if status == "ERROR" or step_count > INEFF_THRESHOLD:
                    logger.info(f"[yellow]Triggering ACE learning (failure/inefficiency)...[/yellow]", extra={"markup": True})
                    task = asyncio.create_task(
                        self._async_learn(domain, status, step_count, exec_time, history)
                    )
                    learning_tasks_for_domain.append(task)
                    self._learning_tasks.append(task)  # Keep for global tracking
                elif status in ["AVAILABLE", "TAKEN"] and step_count <= MAX_STEPS_TARGET:
                    logger.info(f"[green]Caching successful pattern...[/green]", extra={"markup": True})
                    task = asyncio.create_task(
                        self._async_learn_success(domain, step_count, result.site_used, history)
                    )
                    learning_tasks_for_domain.append(task)
                    self._learning_tasks.append(task)  # Keep for global tracking

            if status == "ERROR":
                self.metrics["errors"] += 1
                logger.warning(f"[yellow]✗ Status: {status}[/yellow]", extra={"markup": True})
            else:
                logger.info(f"[green]✓ Status: {status}[/green]", extra={"markup": True})
                # proxy signal that ACE guidance yielded a conclusive result
                self.metrics["ace_improvements_applied"] += 1

            console.print(f"[dim]Steps: {step_count} | Domain: {exec_time}s | Tokens: {tokens_used} | Site: {result.site_used or 'unknown'} | ACE: Active[/dim]")

        except asyncio.TimeoutError:
            msg = "Timeout: Agent took longer than 3 minutes"
            logger.error(f"✗ {msg}")
            result.status = "ERROR"
            result.error_message = msg

            # Capture any steps/tokens used before timeout
            if 'history' in locals() and history is not None:
                result.steps = self._extract_step_count(history)
                result.site_used = self._detect_site_used(history)
                # Try to capture tokens from history
                if hasattr(history, 'usage') and history.usage:
                    result.tokens_used = history.usage.total_tokens

            self.metrics["errors"] += 1

        except Exception as e:
            msg = f"Unexpected error: {e}"
            logger.error(f"✗ {msg}", exc_info=True)
            result.status = "ERROR"
            result.error_message = msg

            # Capture any steps/tokens used before the error
            if 'history' in locals() and history is not None:
                result.steps = self._extract_step_count(history)
                result.site_used = self._detect_site_used(history)
                # Try to capture tokens from history
                if hasattr(history, 'usage') and history.usage:
                    result.tokens_used = history.usage.total_tokens

            self.metrics["errors"] += 1

        finally:
            if browser:
                try:
                    await browser.stop()
                except Exception as e:
                    logger.debug(f"Failed to stop browser: {e}")

        return result

    async def _async_learn(
        self,
        domain: str,
        status: str,
        step_count: int,
        exec_time: float,
        history: Any
    ) -> bool:
        """Learn from failures and inefficiencies."""
        try:
            recent_actions = []
            try:
                recent_actions = (history.action_names() or [])[-5:]
            except Exception:
                pass

            recent_urls = []
            try:
                recent_urls = (history.urls() or [])[-3:]
            except Exception:
                pass

            context = (
                f"Checked {domain}: {status} in {step_count} steps, {exec_time}s. "
                f"Target: <{MAX_STEPS_TARGET} steps. "
                f"Actions: {recent_actions}. "
                f"URLs visited: {recent_urls}. "
            )

            reflection = await asyncio.to_thread(
                self.reflector.reflect,
                question="Why was this domain check inefficient or failed?",
                answer=status,
                context=context,
                playbook=self.playbook,
            )

            delta = await asyncio.to_thread(
                self.curator.curate,
                reflection=reflection,
                playbook=self.playbook,
            )

            if delta and getattr(delta, "operations", None):
                self.playbook.apply_delta(delta)
                logger.info(f"[green]✓ ACE learned new strategy ({len(self.playbook.bullets())} total)[/green]", extra={"markup": True})
                return True
        except Exception as e:
            logger.debug(f"Learning failed: {e}")

        return False

    async def _async_learn_success(
        self,
        domain: str,
        step_count: int,
        site_used: Optional[str],
        history: Any
    ) -> bool:
        """Learn from successful efficient checks."""
        try:
            actions = []
            try:
                actions = history.action_names() or []
            except Exception:
                pass

            context = (
                f"Successfully checked {domain} in {step_count} steps (efficient). "
                f"Site used: {site_used}. "
                f"Actions: {actions}. "
                f"This pattern should be reused."
            )

            reflection = await asyncio.to_thread(
                self.reflector.reflect,
                question="What made this domain check efficient and successful?",
                answer="SUCCESS",
                context=context,
                playbook=self.playbook,
            )

            delta = await asyncio.to_thread(
                self.curator.curate,
                reflection=reflection,
                playbook=self.playbook,
            )

            if delta and getattr(delta, "operations", None):
                self.playbook.apply_delta(delta)
                logger.info(f"[green]✓ ACE learned from success ({len(self.playbook.bullets())} total)[/green]", extra={"markup": True})
                return True
        except Exception as e:
            logger.debug(f"Success learning failed: {e}")

        return False

    def _parse_agent_output(self, output: str, domain: str) -> str:
        if not output:
            return "ERROR"

        out = output.upper()
        d = domain.upper()

        # Exact format (preferred)
        if f"AVAILABLE: {d}" in out:
            return "AVAILABLE"
        if f"TAKEN: {d}" in out:
            return "TAKEN"
        if "ERROR:" in out:
            return "ERROR"

        # Heuristics (degrades gracefully)
        if "AVAILABLE" in out and "NOT AVAILABLE" not in out and "UNAVAILABLE" not in out:
            return "AVAILABLE"
        if any(tok in out for tok in ["TAKEN", "UNAVAILABLE", "NOT AVAILABLE", "ALREADY REGISTERED"]):
            return "TAKEN"

        logger.warning(f"Could not parse output: {output[:120]}")
        return "ERROR"

    async def run_session(self) -> Dict[str, Any]:
        console.print(Panel.fit(
            "[bold white]SESSION 2: WITH ACE LEARNING[/bold white]\n"
            f"[dim]Model: {self.model} | ACE: {self.ace_model} | Learning: Enabled[/dim]\n"
            f"[green]Generator→Reflector→Curator optimizing performance[/green]",
            border_style="green",
            padding=(1, 2),
        ))

        self.load_domains()
        self.metrics["start_time"] = datetime.now().isoformat()
        t0 = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[green]Checking domains with ACE...",
                total=len(self.domains)
            )

            for d in self.domains:
                res = await self.check_single_domain(d)
                self._accumulate_metrics(res)
                progress.advance(task_id)
                await asyncio.sleep(0.5)

        # Wait deterministically for background learning to finish and count updates
        if self._learning_tasks:
            learn_results = await asyncio.gather(*self._learning_tasks, return_exceptions=True)
            applied = sum(1 for r in learn_results if r is True)
            self.metrics["learning_updates_applied"] = applied

        t1 = time.time()
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["total_time_seconds"] = round(t1 - t0, 2)
        self.metrics["domains_checked"] = len(self.domains)

        succ = self.metrics["domains_available"] + self.metrics["domains_taken"]
        self.metrics["accuracy_rate"] = round((succ / max(1, len(self.domains))) * 100, 2)
        self.metrics["average_time_per_domain"] = round(
            self.metrics["total_time_seconds"] / max(1, len(self.domains)), 2
        )
        # Domain time averages
        self.metrics["average_domain_check_time"] = round(
            self.metrics["total_domain_check_time"] / max(1, len(self.domains)), 2
        )
        self.metrics["average_steps_per_domain"] = round(
            self.metrics["total_steps"] / max(1, len(self.domains)), 2
        )
        self.metrics["average_tokens_per_domain"] = round(
            self.metrics["total_tokens"] / max(1, len(self.domains)), 2
        )


        self.persistence.save_playbook(self.playbook)
        out_path = self._save_results()
        self._print_summary()
        self._print_comparison()

        return {
            "metrics": self.metrics,
            "results": [r.to_dict() for r in self.results],
            "output_file": str(out_path),
        }

    def _accumulate_metrics(self, res: DomainCheckResult) -> None:
        self.results.append(res)
        self.metrics["total_steps"] += res.steps
        self.metrics["total_tokens"] += res.tokens_used

        # Domain time tracking
        self.metrics["total_domain_check_time"] += res.domain_check_time_seconds

        if res.status == "AVAILABLE":
            self.metrics["domains_available"] += 1
        elif res.status == "TAKEN":
            self.metrics["domains_taken"] += 1


    def _save_results(self) -> Path:
        data = {
            "metadata": {
                "session": "Session 2 - With ACE Learning",
                "description": "Domain checker with Generator→Reflector→Curator optimization",
                "model": self.model,
                "ace_model": self.ace_model,
                "generated_at": datetime.now().isoformat(),
            },
            "metrics": self.metrics,
            "results": [r.to_dict() for r in self.results],
            "playbook": self.playbook.as_prompt(),
            "learned_strategies": len(self.playbook.bullets()),
        }

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"session2_with_ace_{ts}.json")
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"[green]✓ Results saved to: {path}[/green]", extra={"markup": True})
        return path

    def _print_summary(self) -> None:
        # Per-domain results table
        results_table = Table(title="PER-DOMAIN RESULTS (WITH ACE)", show_header=True,
                             header_style="bold cyan", border_style="blue")
        results_table.add_column("#", style="dim", width=3)
        results_table.add_column("Domain", style="cyan", width=20)
        results_table.add_column("Status", style="yellow", width=10)
        results_table.add_column("Steps", style="green", justify="right", width=6)
        results_table.add_column("Time (s)", style="green", justify="right", width=9)
        results_table.add_column("Tokens", style="blue", justify="right", width=7)
        results_table.add_column("Attempts", style="yellow", justify="right", width=8)
        results_table.add_column("Site Used", style="magenta", width=15)

        for idx, result in enumerate(self.results, 1):
            status_style = "green" if result.status in ["AVAILABLE", "TAKEN"] else "red"
            results_table.add_row(
                str(idx),
                result.domain,
                f"[{status_style}]{result.status}[/{status_style}]",
                str(result.steps),
                f"{result.total_time_seconds:.1f}",
                str(result.tokens_used),
                str(result.attempts),
                result.site_used or "unknown"
            )

        console.print("\n")
        console.print(results_table)
        console.print("\n")

        # Summary metrics table
        t = Table(
            title="SESSION 2 SUMMARY (WITH ACE)",
            show_header=True,
            header_style="bold magenta",
            border_style="green"
        )
        t.add_column("Metric", style="cyan", width=34)
        t.add_column("Value", style="yellow", justify="right")

        t.add_row("Total Time", f"{self.metrics['total_time_seconds']}s")
        t.add_row("Total Domain Check Time", f"{self.metrics['total_domain_check_time']:.1f}s")
        t.add_row("", "")
        t.add_row("Domains Checked", str(self.metrics["domains_checked"]))
        t.add_row("Total Steps", str(self.metrics["total_steps"]))
        t.add_row("Total Tokens", str(self.metrics["total_tokens"]))
        t.add_row("", "")
        t.add_row("Avg Steps/Domain", f"{self.metrics['average_steps_per_domain']:.1f}")
        t.add_row("Avg Time/Domain", f"{self.metrics['average_time_per_domain']}s")
        t.add_row("Avg Domain Check Time", f"{self.metrics['average_domain_check_time']:.1f}s")
        t.add_row("Avg Tokens/Domain", f"{self.metrics['average_tokens_per_domain']:.1f}")
        t.add_row("", "")
        t.add_row("Available", str(self.metrics["domains_available"]))
        t.add_row("Taken", str(self.metrics["domains_taken"]))
        t.add_row("Errors", str(self.metrics["errors"]))
        t.add_row("", "")
        t.add_row("Accuracy Rate", f"{self.metrics['accuracy_rate']}%")
        t.add_row("", "")
        t.add_row("ACE Improvements Applied", str(self.metrics["ace_improvements_applied"]))
        t.add_row("Learning Updates Applied", str(self.metrics["learning_updates_applied"]))

        console.print("\n")
        console.print(t)
        console.print("\n")

    def _print_comparison(self) -> None:
        """Compare with Session 1 baseline (robust to missing metrics)."""
        try:
            baseline_files = list(Path(".").glob("session1_baseline_*.json"))
            if not baseline_files:
                return

            latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)
            with open(latest_baseline, "r", encoding="utf-8") as f:
                baseline_data = json.load(f)

            baseline_metrics = baseline_data.get("metrics", {}) or {}
            # Derive missing average_steps_per_domain if needed
            baseline_steps = baseline_metrics.get("average_steps_per_domain")
            if baseline_steps is None:
                results = baseline_data.get("results", []) or []
                if results:
                    steps_sum = sum(int(r.get("steps", 0)) for r in results)
                    baseline_steps = steps_sum / max(1, len(results))
                else:
                    baseline_steps = 0.0

            # Calculate improvements
            ace_steps = self.metrics["average_steps_per_domain"]
            step_improvement = ((baseline_steps - ace_steps) / baseline_steps * 100) if baseline_steps > 0 else 0.0

            baseline_time = baseline_metrics.get("average_time_per_domain", 0.0)
            ace_time = self.metrics["average_time_per_domain"]
            time_improvement = ((baseline_time - ace_time) / baseline_time * 100) if baseline_time > 0 else 0.0

            baseline_accuracy = baseline_metrics.get("accuracy_rate", 0.0)
            ace_accuracy = self.metrics["accuracy_rate"]
            accuracy_improvement = ace_accuracy - baseline_accuracy

            comparison = Table(
                title="COMPARISON: Session 2 (ACE) vs Session 1 (Baseline)",
                show_header=True,
                header_style="bold cyan",
                border_style="cyan"
            )
            comparison.add_column("Metric", style="cyan")
            comparison.add_column("Baseline", style="yellow", justify="right")
            comparison.add_column("With ACE", style="green", justify="right")
            comparison.add_column("Improvement", style="magenta", justify="right")

            comparison.add_row(
                "Avg Steps/Domain",
                f"{baseline_steps:.1f}",
                f"{ace_steps:.1f}",
                f"{step_improvement:+.1f}%"
            )
            comparison.add_row(
                "Avg Time/Domain",
                f"{baseline_time:.1f}s",
                f"{ace_time:.1f}s",
                f"{time_improvement:+.1f}%"
            )
            comparison.add_row(
                "Accuracy Rate",
                f"{baseline_accuracy:.1f}%",
                f"{ace_accuracy:.1f}%",
                f"{accuracy_improvement:+.1f}pp"
            )

            console.print("\n")
            console.print(comparison)
            console.print("\n")

            # Verdict
            if step_improvement > 0 or time_improvement > 0 or accuracy_improvement > 0:
                verdict = "[bold green]✓ ACE SHOWS IMPROVEMENT![/bold green]"
                if step_improvement > 20:
                    verdict += f"\n  [green]• {step_improvement:.0f}% fewer steps[/green]"
                if time_improvement > 20:
                    verdict += f"\n  [green]• {time_improvement:.0f}% faster[/green]"
                if accuracy_improvement > 5:
                    verdict += f"\n  [green]• {accuracy_improvement:.0f}pp better accuracy[/green]"
            else:
                verdict = "[yellow]⚠ No significant improvement yet[/yellow]\n  [dim]May need more samples or parameter tuning[/dim]"

            console.print(Panel.fit(verdict, border_style="cyan", padding=(1, 2)))

        except Exception as e:
            logger.debug(f"Could not generate comparison: {e}")


# ----------------------------
# Entrypoint
# ----------------------------

async def main():
    try:
        checker = DomainCheckerWithACE(
            domains_file="domains.txt",
            model="gpt-4o",
            ace_model="gpt-4o-mini",
            headless=False,
            max_retries=2,
        )

        results = await checker.run_session()

        logger.info("[bold green]✓ Session 2 (With ACE) completed![/bold green]", extra={"markup": True})
        return results

    except KeyboardInterrupt:
        logger.warning("\n⚠ Session interrupted by user")
        sys.exit(1)

    except Exception as e:
        logger.error(f"[bold red]✗ Session failed: {e}[/bold red]", extra={"markup": True})
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
