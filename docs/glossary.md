# Financial Glossary

This glossary explains the financial terms Nova Trader currently uses or is likely to use in the first recommendation-engine cut. The goal is practical understanding: what the term means, why it matters, and how it shows up in this codebase.

This is not investment advice. Treat the definitions as working product vocabulary for research, recommendation structure, and engineering decisions.

## Product And Recommendation Terms

- **Recommendation engine**: A system that converts a user question into data gathering, analysis, risk checks, and a structured recommendation.
- **Query router**: The component that reads the user's question and classifies the intent, such as single-stock recommendation, portfolio review, valuation, risk exposure, backtest, or news sentiment.
- **Workflow selector**: The component that maps a routed query to the analysis path that should run.
- **Ticker**: The short market symbol for a traded security, such as `AAPL` or `MSFT`.
- **Analyst agent**: A specialized analysis module. In Nova Trader, agents can represent technical analysis, fundamentals, valuation, news sentiment, risk, portfolio logic, or investor-style personas.
- **Signal**: A simplified directional opinion from an agent, usually `bullish`, `bearish`, or `neutral`.
- **Bullish**: The agent thinks the asset has positive upside or attractive conditions.
- **Bearish**: The agent thinks the asset has downside risk or unattractive conditions.
- **Neutral**: The agent does not see enough directional edge, or bullish and bearish evidence roughly balance.
- **Hedge**: A position intended to offset part of the risk in another position.
- **Equity long/short**: A strategy that owns selected stocks long and shorts other stocks to reduce broad market exposure.
- **Hedged recommendation**: A recommendation where an opening long has a corresponding short candidate instead of standing alone.
- **Confidence**: How strongly an agent or recommendation believes its signal. In this repo it appears as either `0.0-1.0` or `0-100`, depending on the layer.
- **Conviction**: A coarser label for recommendation strength, such as low, medium, or high.
- **Evidence**: A structured fact used to support a recommendation, such as valuation gap, margin trend, risk limit, sentiment count, or price indicator.
- **Horizon**: The time frame of the recommendation, such as intraday, short term, medium term, or long term.
- **What would change our mind**: A list of conditions that would weaken or reverse the recommendation. This is important for making recommendations falsifiable.

## Market Data Terms

- **Open price**: The first traded price for a period, usually a trading day.
- **High price**: The highest traded price during the period.
- **Low price**: The lowest traded price during the period.
- **Close price**: The final traded price during the period. Many indicators use close price because it is stable and widely reported.
- **OHLC**: Open, high, low, close. A standard price-data format.
- **Volume**: The number of shares traded during a period. High volume can confirm that a price move has broad participation.
- **Return**: The percentage gain or loss over a period. Example: if price moves from 100 to 105, return is 5%.
- **Daily return**: The day-over-day percentage change in price or portfolio value.
- **Market capitalization / market cap**: The total market value of a company's equity. Roughly `share price * shares outstanding`.
- **Enterprise value / EV**: A company's total business value, often approximated as `market cap + debt - cash`. It is useful when comparing companies with different debt levels.

## Technical Analysis Terms

- **Technical analysis**: Analysis based on price, volume, trends, volatility, and market behavior rather than financial statements.
- **Moving average**: An average of recent prices over a rolling window. It smooths noisy price action so trends are easier to see.
- **Simple moving average / SMA**: A moving average where each price in the window has equal weight.
- **Exponential moving average / EMA**: A moving average that weights recent prices more heavily. Nova Trader uses EMAs in the technical trend agent.
- **Trend following**: A strategy that tries to ride persistent price direction. In Nova Trader, short EMA above medium EMA above long EMA is treated as bullish trend evidence.
- **Mean reversion**: The idea that prices can move too far from a normal level and later return closer to average.
- **Momentum**: The strength and persistence of price movement. Positive momentum means price has been rising over recent windows.
- **Volume momentum**: Current volume compared with average volume. It helps check whether price momentum has confirmation from trading activity.
- **Relative Strength Index / RSI**: A momentum indicator usually shown from 0 to 100. High RSI can suggest overbought conditions; low RSI can suggest oversold conditions.
- **Bollinger Bands**: Bands placed above and below a moving average, usually based on standard deviation. They help identify unusually high or low prices relative to recent volatility.
- **Z-score**: How far a value is from its average in standard-deviation units. A price z-score below -2 can indicate unusually depressed price action; above +2 can indicate unusually stretched price action.
- **Average Directional Index / ADX**: A trend-strength indicator. It measures trend strength, not whether the trend is up or down.
- **Average True Range / ATR**: A volatility measure based on the trading range of each period. Nova Trader uses an ATR ratio to reason about volatility relative to price.
- **Historical volatility**: Volatility calculated from past returns.
- **Annualized volatility**: Daily volatility scaled to a yearly estimate, often using roughly 252 trading days.
- **Volatility regime**: Whether current volatility is low, normal, or high compared with recent history.
- **Volatility z-score**: How unusual current volatility is compared with its own recent average.
- **Skewness**: Whether return distribution has a longer positive or negative tail.
- **Kurtosis**: How extreme the tails of return distribution are compared with a normal distribution.
- **Hurst exponent**: A statistic used to estimate whether a time series is trending, mean reverting, or close to random walk. In this repo, values below 0.5 are treated as more mean reverting.
- **Statistical arbitrage / stat-arb**: A strategy family that looks for statistical price relationships or mean-reversion behavior. Nova Trader currently uses simple price-action statistics rather than a full pairs-trading engine.

## Fundamental Metrics

- **Fundamental analysis**: Analysis based on a company's financial statements, business quality, growth, margins, balance sheet, and cash flow.
- **Revenue / sales**: Money a company earns from selling products or services before expenses.
- **Revenue growth**: How fast revenue is increasing or decreasing.
- **Net income / earnings**: Profit after expenses, taxes, interest, and other costs.
- **Earnings growth**: How fast earnings are increasing or decreasing.
- **Earnings per share / EPS**: Earnings divided by shares outstanding. It shows how much profit belongs to each share.
- **Free cash flow / FCF**: Cash generated after operating expenses and capital expenditures. It is often central to valuation because it represents cash a business can potentially return or reinvest.
- **Free cash flow per share**: FCF divided by shares outstanding.
- **Free cash flow growth**: Growth in FCF over time.
- **Capital expenditure / capex**: Money spent on long-term assets such as equipment, infrastructure, or property.
- **Depreciation and amortization**: Accounting expenses that spread the cost of assets over time.
- **Working capital**: Short-term operating capital, often current assets minus current liabilities.
- **Gross margin**: Gross profit divided by revenue. It measures how much revenue remains after direct production costs.
- **Operating margin**: Operating income divided by revenue. It measures profitability after operating expenses.
- **Net margin**: Net income divided by revenue. It measures final profit after all expenses.
- **Return on equity / ROE**: Net income divided by shareholder equity. It measures profitability relative to owner capital.
- **Return on assets / ROA**: Net income divided by assets. It measures how efficiently assets generate profit.
- **Return on invested capital / ROIC**: Profitability relative to capital invested in the business. Strong, consistent ROIC can indicate a high-quality business.
- **Asset turnover**: Revenue divided by assets. It measures how efficiently assets produce sales.
- **Inventory turnover**: How quickly inventory is sold and replaced.
- **Receivables turnover**: How quickly customers pay what they owe.
- **Days sales outstanding / DSO**: Average number of days it takes to collect payment after a sale.
- **Operating cycle**: Time needed to buy inventory, sell it, and collect cash.
- **Current ratio**: Current assets divided by current liabilities. A liquidity measure for near-term obligations.
- **Quick ratio**: Similar to current ratio but excludes inventory. It is a stricter liquidity measure.
- **Cash ratio**: Cash and cash equivalents divided by current liabilities. It is the strictest common liquidity measure.
- **Debt-to-equity / D/E**: Debt divided by shareholder equity. It measures leverage.
- **Debt-to-assets**: Debt divided by total assets. It measures how much of the asset base is financed by debt.
- **Interest coverage**: Operating income or EBIT divided by interest expense. It shows how comfortably a company can pay interest.
- **Book value**: Accounting value of shareholders' equity.
- **Book value per share**: Book value divided by shares outstanding.
- **Book value growth**: Growth in book value over time.
- **Share dilution**: Increase in shares outstanding. Dilution can reduce each existing shareholder's ownership percentage.
- **Buyback**: A company repurchasing its own shares. Buybacks can reduce share count if they exceed new share issuance.
- **Payout ratio**: Percentage of earnings paid out as dividends.
- **Insider activity**: Buying or selling by company insiders, such as executives or directors. Insider buying can be a useful signal, but it is not automatically bullish.
- **News sentiment**: Classification of news as positive, negative, or neutral for a stock.
- **Moat**: A durable competitive advantage that can protect profits over time.
- **Pricing power**: A company's ability to raise prices without losing too much customer demand.
- **Capital intensity**: How much capex is needed to maintain or grow the business. Asset-light businesses usually have lower capital intensity.

## Valuation Terms

- **Valuation**: Estimating what a business or stock is worth.
- **Intrinsic value**: An estimate of a business's fair value based on fundamentals, cash flows, and assumptions.
- **Valuation gap**: Difference between estimated intrinsic value and current market value.
- **Margin of safety**: Extra discount between estimated value and market price. Example: if value is estimated at 100 and price is 70, the margin of safety is 30%.
- **Discounted cash flow / DCF**: A valuation method that estimates future cash flows and discounts them back to today's value.
- **Discount rate**: The required return used to convert future cash flows into present value.
- **Weighted average cost of capital / WACC**: A blended cost of equity and debt capital. Nova Trader uses WACC in the enhanced DCF valuation logic.
- **Terminal value**: Estimated value of a business beyond the explicit forecast period in a DCF.
- **Terminal multiple**: A multiple applied to a future financial metric to estimate terminal value.
- **Owner earnings**: A Buffett-style measure approximating cash earnings available to owners: net income plus depreciation/amortization minus capex and working-capital needs.
- **Residual income model**: A valuation approach based on book value plus future income above the required return on equity.
- **Comparable analysis / comps**: Valuation by comparing a company against similar companies using ratios such as P/E, P/S, or EV/EBITDA.
- **Price-to-earnings / P/E ratio**: Share price divided by EPS, or market cap divided by earnings. A high P/E can mean high growth expectations or overvaluation; a low P/E can mean cheapness or business weakness.
- **Price-to-book / P/B ratio**: Market price divided by book value per share.
- **Price-to-sales / P/S ratio**: Market price divided by sales per share, or market cap divided by revenue.
- **PEG ratio**: P/E ratio divided by expected earnings growth. It tries to judge valuation relative to growth.
- **EV/EBITDA**: Enterprise value divided by EBITDA. Useful for comparing operating businesses with different debt levels.
- **EV/revenue**: Enterprise value divided by revenue. Often used when earnings are low, negative, or distorted.
- **Free cash flow yield**: FCF divided by market cap or enterprise value. Higher yield can indicate cheaper valuation, assuming cash flow is durable.
- **EBIT**: Earnings before interest and taxes.
- **EBITDA**: Earnings before interest, taxes, depreciation, and amortization.

## Portfolio, Risk, And Backtesting Terms

- **Portfolio**: A collection of positions plus cash.
- **Position**: The amount held in a specific asset.
- **Long position**: Owning an asset. It benefits if price rises.
- **Short position**: Borrowing and selling an asset with the goal of buying it back cheaper later. It benefits if price falls and can lose heavily if price rises.
- **Pair trade**: A long position and short position chosen together. The goal is to express relative preference while reducing broad market exposure.
- **Hedge ratio**: Short notional divided by long notional. A ratio near 1.0 means the short side is roughly dollar-matched to the long side.
- **Cover**: Closing a short position by buying shares back.
- **Cash balance**: Uninvested cash in the portfolio.
- **Equity / net liquidation value**: Approximate total portfolio value after marking positions to current prices.
- **Margin**: Borrowed capital or collateral mechanics used to increase exposure or support short positions.
- **Margin requirement**: Required collateral for a margin or short position.
- **Position sizing**: Deciding how large a trade or holding should be.
- **Risk limit**: A maximum allowed position size or exposure.
- **Volatility-adjusted limit**: A risk limit that shrinks when an asset is more volatile and expands when it is more stable.
- **Correlation**: How closely two assets move together. High positive correlation means they often move in the same direction.
- **Correlation multiplier**: Nova Trader's adjustment that reduces position limits when a ticker is highly correlated with existing exposure.
- **Long exposure**: Dollar value of long positions.
- **Short exposure**: Dollar value of short positions.
- **Gross exposure**: Long exposure plus short exposure.
- **Net exposure**: Long exposure minus short exposure.
- **Long/short ratio**: Long exposure divided by short exposure.
- **Backtest**: A historical simulation of how a strategy or recommendation process would have performed.
- **Equity curve**: Portfolio value over time during a backtest.
- **Benchmark**: A reference used for comparison, such as an index or a baseline strategy.
- **Sharpe ratio**: Risk-adjusted return using total volatility. Higher is generally better, but it can hide downside-specific risk.
- **Sortino ratio**: Risk-adjusted return using downside volatility. It focuses more directly on harmful volatility.
- **Max drawdown**: The largest peak-to-trough decline in portfolio value during a period.
- **Risk-free rate**: A baseline return used when calculating excess returns, often approximated with Treasury yields.
- **Annual trading days**: The number of trading days assumed in a year, commonly around 252 for U.S. equities.

## External Projects And Resources Worth Studying

These are useful references for deciding how Nova Trader should evolve. They should inform design, not dictate the architecture.

- **[ai-hedge-fund](https://github.com/virattt/ai-hedge-fund)**: An open-source multi-agent investment research project with investor-style agents, risk management, and portfolio-management concepts. Strong reference for product shape, but Nova Trader should avoid becoming only a persona demo.
- **[TradingAgents](https://github.com/TauricResearch/TradingAgents)**: A multi-agent LLM trading framework. Useful for studying agent roles, debate/research flows, persistent decision logs, structured outputs, and report generation.
- **[FinRobot](https://github.com/AI4Finance-Foundation/FinRobot)**: An AI4Finance project focused on financial agents and LLM workflows. Useful for seeing how financial-agent tasks are decomposed across research, valuation, risk, and automation.
- **[OpenBB Platform](https://github.com/OpenBB-finance/OpenBB)**: Open-source investment research and data tooling. Useful as inspiration for data access, analyst workflows, and user-facing research ergonomics.
- **[SEC Investor.gov glossary](https://www.investor.gov/introduction-investing/investing-basics/glossary)**: Good source for beginner-safe definitions of common investing terms.
- **FINRA investor education/glossary**: Good for risk, portfolio, margin, and investor-protection terminology.
- **Corporate Finance Institute / valuation resources**: Useful for DCF, WACC, enterprise value, and accounting vocabulary.
- **Fidelity, Schwab, and Investopedia technical-analysis explainers**: Useful for practical explanations of moving averages, RSI, MACD, Bollinger Bands, and related technical indicators.

## Terms To Add Soon

- **Alpha**: Excess return over a benchmark after accounting for risk.
- **Beta**: Sensitivity to market movements.
- **Value at Risk / VaR**: Estimate of possible loss over a time window at a confidence level.
- **Expected shortfall**: Average loss in the worst-case tail beyond VaR.
- **Factor exposure**: Exposure to systematic drivers such as value, growth, momentum, quality, size, or sector.
- **Liquidity risk**: Risk that a position cannot be entered or exited without moving the price.
- **Slippage**: Difference between expected trade price and executed trade price.
- **Transaction cost**: Broker fees, spreads, market impact, and other trading costs.
- **MACD**: Moving Average Convergence Divergence, a momentum/trend indicator based on moving averages. It is not currently central in Nova Trader but may be useful later.
