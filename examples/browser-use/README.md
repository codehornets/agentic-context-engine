# ACE + Browser-Use Domain Checker Demo

This directory contains two implementations of a domain availability checker to demonstrate ACE's learning capabilities:

## 🤖 Baseline (No Learning)
**File:** `baseline_domain_checker.py`

- Uses browser automation to check domain availability
- **No learning** - same approach every time
- May repeat the same mistakes
- Performance stays constant

```bash
python examples/browser-use/baseline_domain_checker.py
```

## 🧠 With ACE Learning
**File:** `ace_domain_checker.py`

- Uses the same browser automation
- **Learns after each domain check** using OnlineAdapter
- Builds strategies in a Playbook
- Performance improves over time
- Avoids repeating failed approaches

```bash
python examples/browser-use/ace_domain_checker.py
```

## 🔬 Key Differences

| Aspect | Baseline | With ACE |
|--------|----------|----------|
| **Learning** | ❌ None | ✅ After each domain |
| **Strategy** | 🔒 Fixed | 📈 Evolving |
| **Mistakes** | 🔄 Repeats | 🚫 Avoids |
| **Performance** | ➡️ Constant | 📈 Improves |
| **Memory** | 💭 None | 📚 Playbook |

## 📊 Expected Results

**Baseline:** Consistent performance, may hit the same issues repeatedly

**ACE:**
- Early domains may have more steps/errors
- Later domains should be more efficient
- Learns which sites work best
- Develops strategies to avoid CAPTCHAs
- Saves learned strategies to `ace_domain_playbook.json`

## 🚀 Running the Comparison

1. **Install dependencies:**
   ```bash
   uv sync --extra demos
   ```

2. **Set up environment:**
   ```bash
   export OPENAI_API_KEY="your-key-here"
   ```

3. **Run both demos:**
   ```bash
   # First run baseline (no learning)
   python examples/browser-use/baseline_domain_checker.py

   # Then run ACE version (with learning)
   python examples/browser-use/ace_domain_checker.py
   ```

4. **Compare results:**
   - Look at step counts and success rates
   - Notice if ACE improves efficiency over time
   - Check the saved playbook for learned strategies

## 🎯 What ACE Demonstrates

- **Incremental Learning:** Each domain check teaches ACE something new
- **Strategy Evolution:** Playbook grows with successful patterns
- **Error Avoidance:** Learns to avoid sites with CAPTCHAs or issues
- **Efficiency Gains:** Fewer steps needed as it learns optimal paths
- **Knowledge Persistence:** Saves learned strategies for future use

This showcases ACE's core value: **turning experience into improved performance**.