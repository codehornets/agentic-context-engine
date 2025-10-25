#!/usr/bin/env python3
"""
Baseline Domain Checker (WITHOUT ACE)

Simple domain checker using browser automation without any learning.
Compare this with ace_domain_checker.py to see ACE's value.
"""

import asyncio
from pathlib import Path
from typing import List
from dotenv import load_dotenv

from browser_use import Agent, Browser, ChatOpenAI

load_dotenv()


def get_test_domains() -> List[str]:
    """Get list of test domains to check."""
    return [
        "test-domain-12345.com",
        "example-test-9999.org",
        "mytest-domain-xyz.net"
    ]


async def check_domain(domain: str, model: str = "gpt-4o-mini", headless: bool = True):
    """Check domain availability without any learning."""
    browser = None
    try:
        # Start browser
        browser = Browser(headless=headless)
        await browser.start()

        # Create agent with basic task (no learning, no strategy optimization)
        llm = ChatOpenAI(model=model, temperature=0.0)

        task = f"""Check if the domain "{domain}" is available for registration.

Use domain lookup websites. Avoid sites with CAPTCHAs.

Output format (exactly one of these):
AVAILABLE: {domain}
TAKEN: {domain}
ERROR: <reason>"""

        agent = Agent(
            task=task,
            llm=llm,
            browser=browser,
            max_actions_per_step=5,
            max_steps=12,
        )

        # Run with timeout
        history = await asyncio.wait_for(agent.run(), timeout=90.0)

        # Parse result
        output = history.final_result() if hasattr(history, "final_result") else ""
        steps = len(history.action_names()) if hasattr(history, "action_names") and history.action_names() else 0

        # Determine status
        status = "ERROR"
        output_upper = output.upper()
        domain_upper = domain.upper()

        if f"AVAILABLE: {domain_upper}" in output_upper:
            status = "AVAILABLE"
        elif f"TAKEN: {domain_upper}" in output_upper:
            status = "TAKEN"

        return {
            "domain": domain,
            "status": status,
            "steps": steps,
            "output": output,
            "success": status != "ERROR"
        }

    except asyncio.TimeoutError:
        return {"domain": domain, "status": "ERROR", "steps": 999, "error": "Timeout", "success": False}
    except Exception as e:
        return {"domain": domain, "status": "ERROR", "steps": 999, "error": str(e), "success": False}
    finally:
        if browser:
            try:
                await browser.stop()
            except:
                pass


def main():
    """Main function - basic domain checking without learning."""

    print("\n🤖 Baseline Domain Checker (WITHOUT ACE)")
    print("🚫 No learning - same approach every time")
    print("=" * 50)

    # Get test domains
    domains = get_test_domains()
    print(f"📋 Testing {len(domains)} domains:")
    for i, domain in enumerate(domains, 1):
        print(f"  {i}. {domain}")

    print("\n🔄 Starting domain checks (no learning)...\n")

    results = []

    # Check each domain without any learning
    for i, domain in enumerate(domains, 1):
        print(f"🔍 [{i}/{len(domains)}] Checking domain: {domain}")

        # Run check
        result = asyncio.run(check_domain(domain, headless=False))
        results.append(result)

        # Show what happened
        status = result['status']
        steps = result['steps']
        success = result['success']

        print(f"   📊 Result: {status} ({'✓' if success else '✗'}) in {steps} steps")
        print()

    # Show final results
    print("=" * 50)
    print("📊 Results:")

    for i, result in enumerate(results, 1):
        domain = result['domain']
        status = result['status']
        steps = result['steps']
        success = result['success']
        print(f"[{i}] {domain}: {status} ({'✓' if success else '✗'}) - {steps} steps")

    # Summary
    successful = sum(1 for r in results if r['success'])
    total_steps = sum(r['steps'] for r in results)
    avg_steps = total_steps / len(results) if results else 0

    print(f"\n✅ Success rate: {successful}/{len(results)} ({100*successful/len(results):.1f}%)")
    print(f"⚡ Average steps: {avg_steps:.1f}")
    print(f"🚫 No learning - same performance every time")

    print(f"\n💡 Compare with: python examples/browser-use/ace_domain_checker.py")
    print(f"   ACE learns and improves after each domain check!")


if __name__ == "__main__":
    main()