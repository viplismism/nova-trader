# Nova Trader Demo Script

## Setup

```bash
nova web --port 8000
```

Open the browser at the printed local URL. For now, use `MiniMax` / `MiniMax-M2.7` when OpenAI quota is exhausted; Nova will also fall back to MiniMax for quota-style provider failures when credentials are configured.

## Flow

1. Run `AAPL` with the adaptive research preset when you want the Dinesh-style research loop, or core council for the faster deterministic path.
2. Watch the activity rows finish: snapshot, technical, fundamentals, sentiment, web research, SEC filings, risk, and portfolio.
3. Click `snapshot [TICKER]` in the inspector to show the fetch log, including web search and SEC filings retrieval.
4. Click individual agents in the inspector to show their activity, signal reasoning, evidence, and LLM attempt metadata.
5. Open the signal card and point out action, confidence, vote mix, risk limit, filing citations, and the analyst-by-analyst reasoning.
6. Ask: `What did the agents think, and what would change the call?`
7. Ask: `What are the main risks with this recommendation?`
8. Optional second run: `AAPL,NVDA` to show multi-ticker cards and portfolio-mode behavior.

## Talk Track

Nova Trader is a council workflow, not a black-box chatbot. The package ingests ticker data, live web research, and SEC filing excerpts once, hands typed views to each analyst, combines those signals, applies risk constraints, and records every run artifact. The UI is intentionally thin: it streams activity, shows the structured signal cards, and lets the user ask follow-up questions grounded in the run that just finished.

The adaptive research preset adds one slower, more agentic analyst: MiniMax first writes focused web and filing queries, Nova executes those searches, then MiniMax synthesizes the cited evidence into a signal. That gives the demo a research-pod feel without depending on Anthropic's hosted web-search credits.

For the MVP, the goal is fast feedback on that loop. We are not trying to ship broker execution, custom quant runners, or enterprise integrations in the first pass. The next useful feedback is whether the signal cards, reasoning, and risk framing match how a real investment team wants to inspect an idea.
