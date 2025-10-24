"""
Session 1: Baseline Domain Availability Checker
WITHOUT ACE — Establishes baseline metrics with a fixed, multi-site pipeline
aligned to Session 2's sites (for fair comparison).

Author: Enterprise ML/AI Team
Version: 1.1.0
"""

import asyncio
import json
import logging
import sys
import time
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

# ----------------------------
# Constants / configuration
# ----------------------------

# No hardcoded site preferences - let agents discover best sites autonomously

MAX_STEPS = 15
AGENT_TIMEOUT_SEC = 180.0  # keep tighter than 5 min
SLEEP_BETWEEN_DOMAINS_SEC = 1.0

# Console & logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%Y-%m-%d %H:%M:%S]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)
load_dotenv()


class DomainCheckResult:
    """
    Result of a domain availability check.
    """

    def __init__(
        self,
        domain: str,
        status: str = 'ERROR',
        steps: int = 0,
        time_seconds: float = 0.0,
        error_message: Optional[str] = None,
        agent_history: Optional[Any] = None,
        site_used: Optional[str] = None,
        tokens_used: int = 0,
        attempts: int = 1,
    ):
        self.domain = domain
        self.status = status
        self.steps = steps
        self.time_seconds = time_seconds
        self.timestamp = datetime.now().isoformat()
        self.error_message = error_message
        self.agent_history = agent_history
        self.site_used = site_used
        self.tokens_used = tokens_used
        self.attempts = attempts

    def to_dict(self) -> Dict[str, Any]:
        return {
            'domain': self.domain,
            'status': self.status,
            'steps': self.steps,
            'time_seconds': self.time_seconds,
            'timestamp': self.timestamp,
            'error_message': self.error_message,
            'site_used': self.site_used,
            'tokens_used': self.tokens_used,
            'attempts': self.attempts,
        }


class DomainCheckerBaseline:
    """
    Baseline checker with a fixed, robust multi-site pipeline (no ACE).
    """

    def __init__(
        self,
        domains_file: str = 'domains.txt',
        model: str = 'gpt-4o',
        max_retries: int = 2,
        headless: bool = False
    ):
        self.domains_file = Path(domains_file)
        self.model = model
        self.max_retries = max_retries
        self.headless = headless

        self.domains: List[str] = []
        self.results: List[DomainCheckResult] = []

        self.metrics: Dict[str, Any] = {
            'session': 1,
            'session_type': 'baseline_autonomous_discovery',
            'start_time': None,
            'end_time': None,
            'total_time_seconds': 0.0,
            'domains_checked': 0,
            'domains_available': 0,
            'domains_taken': 0,
            'errors': 0,
            'total_steps': 0,
            'total_tokens': 0,
            'accuracy_rate': 0.0,
            'average_time_per_domain': 0.0,
            'average_steps_per_domain': 0.0,
            'average_tokens_per_domain': 0.0,
            'model': model,
            'discovery_mode': 'autonomous_site_selection'
        }

        logger.info(f"[bold green]Initialized DomainCheckerBaseline[/bold green]", extra={"markup": True})
        logger.info(f"Model: {model}, Headless: {headless}")

    def load_domains(self) -> List[str]:
        if not self.domains_file.exists():
            raise FileNotFoundError(f"Domains file not found: {self.domains_file}")

        with open(self.domains_file, 'r', encoding='utf-8') as f:
            domains = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        if not domains:
            raise ValueError(f"No valid domains found in {self.domains_file}")

        self.domains = domains
        logger.info(f"✓ Loaded {len(domains)} domains from {self.domains_file}")

        table = Table(title="Loaded Domains", show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim", width=4)
        table.add_column("Domain", style="cyan")
        for idx, domain in enumerate(domains, 1):
            table.add_row(str(idx), domain)
        console.print(table)
        return domains

    def _create_task_prompt(self, domain: str) -> str:
        """
        Autonomous discovery prompt - no hardcoded site preferences.
        """
        return f"""
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

    def _parse_agent_output(self, output: str, domain: str) -> str:
        if not output:
            return 'ERROR'
        out = output.upper()
        d = domain.upper()

        # Exact preferred format
        if f"AVAILABLE: {d}" in out:
            return 'AVAILABLE'
        if f"TAKEN: {d}" in out:
            return 'TAKEN'
        if "ERROR:" in out:
            return 'ERROR'

        # Heuristic fallback
        if "AVAILABLE" in out and "NOT AVAILABLE" not in out and "UNAVAILABLE" not in out:
            return 'AVAILABLE'
        if any(tok in out for tok in ["TAKEN", "UNAVAILABLE", "NOT AVAILABLE", "ALREADY REGISTERED"]):
            return 'TAKEN'
        logger.warning(f"Could not parse agent output: {output[:120]}")
        return 'ERROR'

    def _extract_step_count(self, history: Any) -> int:
        try:
            if hasattr(history, "action_names"):
                names = history.action_names()
                return len(names) if names is not None else 0
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

    async def check_single_domain(self, domain: str) -> DomainCheckResult:
        """
        Check domain with proper time tracking across all retry attempts.
        """
        overall_start_time = time.time()
        attempt = 1
        final_result = DomainCheckResult(domain=domain)
        total_steps = 0
        total_tokens = 0

        while attempt <= self.max_retries + 1:
            attempt_result = await self._check_single_attempt(domain, attempt)

            # Accumulate steps and tokens across all attempts
            total_steps += attempt_result.steps
            total_tokens += attempt_result.tokens_used

            # Always update the final result with the latest attempt data
            final_result.status = attempt_result.status
            final_result.steps = total_steps  # Total steps across all attempts
            final_result.error_message = attempt_result.error_message
            final_result.agent_history = attempt_result.agent_history
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

        # Calculate total time across all attempts
        final_result.time_seconds = round(time.time() - overall_start_time, 2)

        return final_result

    async def _check_single_attempt(self, domain: str, attempt: int = 1) -> DomainCheckResult:
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]Checking:[/bold cyan] [yellow]{domain}[/yellow] (Attempt {attempt}/{self.max_retries + 1})")
        console.print(f"{'='*70}")

        browser = None
        result = DomainCheckResult(domain=domain, attempts=attempt)

        try:
            logger.info(f"Initializing browser for {domain}...")
            browser = Browser(headless=self.headless)
            await browser.start()

            llm = ChatOpenAI(
                model=self.model,
                temperature=0.0,     # deterministic; tighter behavior
                max_retries=5,
                timeout=AGENT_TIMEOUT_SEC,
            )


            task = self._create_task_prompt(domain)

            agent = Agent(
                task=task,
                llm=llm,
                browser=browser,
                max_failures=3,
                max_steps=MAX_STEPS,
            )

            logger.info(f"Starting agent execution for {domain}...")
            t0 = time.time()
            history = await asyncio.wait_for(agent.run(), timeout=AGENT_TIMEOUT_SEC)
            exec_time = round(time.time() - t0, 2)

            step_count = self._extract_step_count(history)
            final_result = history.final_result() if hasattr(history, "final_result") else ""

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

            logger.info(f"Agent completed in {exec_time}s with {step_count} steps, {tokens_used} tokens")
            logger.debug(f"Final result: {final_result}")

            status = self._parse_agent_output(final_result or "", domain)

            result.status = status
            result.steps = step_count
            result.time_seconds = exec_time
            result.agent_history = history
            result.site_used = self._detect_site_used(history)
            result.tokens_used = tokens_used

            if status == 'ERROR':
                logger.warning(f"[yellow]✗ Status: {status}[/yellow]", extra={"markup": True})
                self.metrics['errors'] += 1
            else:
                logger.info(f"[green]✓ Status: {status}[/green]", extra={"markup": True})

            # Show quick line incl. site used and tokens
            site_note = f" | Site: {result.site_used}" if result.site_used else ""
            console.print(f"[dim]Steps: {step_count} | Time: {exec_time}s | Tokens: {tokens_used}{site_note}[/dim]")

        except asyncio.TimeoutError:
            error_msg = "Timeout: Agent took longer than allotted time"
            logger.error(f"✗ {error_msg}")
            result.status = 'ERROR'
            result.error_message = error_msg

            # Capture any steps/tokens used before timeout
            if 'history' in locals() and history is not None:
                result.steps = self._extract_step_count(history)
                result.site_used = self._detect_site_used(history)
                # Try to capture tokens from history
                if hasattr(history, 'usage') and history.usage:
                    result.tokens_used = history.usage.total_tokens

            self.metrics['errors'] += 1

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"✗ {error_msg}", exc_info=True)
            result.status = 'ERROR'
            result.error_message = error_msg

            # Capture any steps/tokens used before the error
            if 'history' in locals() and history is not None:
                result.steps = self._extract_step_count(history)
                result.site_used = self._detect_site_used(history)
                # Try to capture tokens from history
                if hasattr(history, 'usage') and history.usage:
                    result.tokens_used = history.usage.total_tokens

            self.metrics['errors'] += 1

        finally:
            if browser:
                try:
                    await browser.stop()
                    logger.debug(f"Browser stopped for {domain}")
                except Exception as e:
                    logger.warning(f"Failed to stop browser cleanly: {e}")

        return result

    async def run_session(self) -> Dict[str, Any]:
        console.print(Panel.fit(
            "[bold white]SESSION 1: BASELINE (No ACE, Autonomous Site Discovery)[/bold white]\n"
            f"[dim]Model: {self.model}[/dim]",
            border_style="green",
            padding=(1, 2)
        ))

        try:
            self.load_domains()
            self.metrics['start_time'] = datetime.now().isoformat()
            t0 = time.time()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=console
            ) as progress:

                task = progress.add_task(
                    "[cyan]Checking domains (autonomous discovery)...",
                    total=len(self.domains)
                )

                for domain in self.domains:
                    result = await self.check_single_domain(domain)
                    self.results.append(result)
                    self.metrics['total_steps'] += result.steps
                    self.metrics['total_tokens'] += result.tokens_used

                    if result.status == 'AVAILABLE':
                        self.metrics['domains_available'] += 1
                    elif result.status == 'TAKEN':
                        self.metrics['domains_taken'] += 1

                    progress.advance(task)
                    await asyncio.sleep(SLEEP_BETWEEN_DOMAINS_SEC)

            t1 = time.time()
            self.metrics['end_time'] = datetime.now().isoformat()
            self.metrics['total_time_seconds'] = round(t1 - t0, 2)
            self.metrics['domains_checked'] = len(self.domains)

            successful_checks = self.metrics['domains_available'] + self.metrics['domains_taken']
            self.metrics['accuracy_rate'] = round(
                (successful_checks / len(self.domains)) * 100, 2
            ) if self.domains else 0.0

            self.metrics['average_time_per_domain'] = round(
                self.metrics['total_time_seconds'] / len(self.domains), 2
            ) if self.domains else 0.0

            # Average metrics for comparison
            self.metrics['average_steps_per_domain'] = round(
                self.metrics['total_steps'] / len(self.domains), 2
            ) if self.domains else 0.0

            self.metrics['average_tokens_per_domain'] = round(
                self.metrics['total_tokens'] / len(self.domains), 2
            ) if self.domains else 0.0

            output_path = self.save_results()
            self.print_summary()
            return {
                'metrics': self.metrics,
                'results': [r.to_dict() for r in self.results],
                'output_file': str(output_path)
            }

        except Exception as e:
            logger.error(f"Session execution failed: {e}", exc_info=True)
            raise

    def save_results(self) -> Path:
        try:
            output_data = {
                'metadata': {
                    'session': 'Session 1 - Baseline',
                    'description': 'Domain availability checker WITHOUT ACE (autonomous site discovery)',
                    'model': self.model,
                    'discovery_mode': 'autonomous_site_selection',
                    'generated_at': datetime.now().isoformat()
                },
                'metrics': self.metrics,
                'results': [r.to_dict() for r in self.results]
            }

            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'session1_baseline_{ts}.json'
            output_path = Path(filename)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.info(f"[green]✓ Results saved to: {output_path}[/green]", extra={"markup": True})
            return output_path

        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise

    def print_summary(self) -> None:
        # Per-domain results table
        results_table = Table(title="PER-DOMAIN RESULTS", show_header=True,
                             header_style="bold cyan", border_style="blue")
        results_table.add_column("#", style="dim", width=3)
        results_table.add_column("Domain", style="cyan", width=20)
        results_table.add_column("Status", style="yellow", width=10)
        results_table.add_column("Steps", style="green", justify="right", width=6)
        results_table.add_column("Time (s)", style="green", justify="right", width=8)
        results_table.add_column("Tokens", style="blue", justify="right", width=7)
        results_table.add_column("Attempts", style="bright_yellow", justify="right", width=8)
        results_table.add_column("Site Used", style="magenta", width=15)

        for idx, result in enumerate(self.results, 1):
            status_style = "green" if result.status in ["AVAILABLE", "TAKEN"] else "red"
            results_table.add_row(
                str(idx),
                result.domain,
                f"[{status_style}]{result.status}[/{status_style}]",
                str(result.steps),
                f"{result.time_seconds:.1f}",
                str(result.tokens_used),
                str(result.attempts),
                result.site_used or "unknown"
            )

        console.print("\n")
        console.print(results_table)
        console.print("\n")

        # Summary metrics table
        table = Table(title="SESSION 1 SUMMARY (Autonomous Site Discovery)", show_header=True,
                      header_style="bold magenta", border_style="green")
        table.add_column("Metric", style="cyan", width=34)
        table.add_column("Value", style="yellow", justify="right")

        table.add_row("Total Time", f"{self.metrics['total_time_seconds']}s")
        table.add_row("Domains Checked", str(self.metrics['domains_checked']))
        table.add_row("Total Steps", str(self.metrics['total_steps']))
        table.add_row("Total Tokens", str(self.metrics['total_tokens']))
        table.add_row("Avg Steps/Domain", f"{self.metrics['average_steps_per_domain']}")
        table.add_row("Avg Time/Domain", f"{self.metrics['average_time_per_domain']}s")
        table.add_row("Avg Tokens/Domain", f"{self.metrics['average_tokens_per_domain']}")
        table.add_row("", "")
        table.add_row("Available", str(self.metrics['domains_available']))
        table.add_row("Taken", str(self.metrics['domains_taken']))
        table.add_row("Errors", str(self.metrics['errors']))
        table.add_row("", "")
        table.add_row("Accuracy Rate", f"{self.metrics['accuracy_rate']}%")

        console.print("\n")
        console.print(table)
        console.print("\n")


async def main():
    try:
        checker = DomainCheckerBaseline(
            domains_file='domains.txt',
            model='gpt-4o',
            max_retries=2,
            headless=False  # True for CI/automation
        )
        results = await checker.run_session()
        logger.info("[bold green]✓ Session 1 completed successfully![/bold green]", extra={"markup": True})
        return results

    except KeyboardInterrupt:
        logger.warning("\n⚠ Session interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"[bold red]✗ Session failed: {e}[/bold red]", extra={"markup": True})
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
