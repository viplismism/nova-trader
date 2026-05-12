from src.router import DataModule, QueryIntent, QueryRoute, TimeHorizon


def test_query_route_normalizes_tickers_and_modules():
    route = QueryRoute(
        raw_query="Should we buy nvda?",
        intent=QueryIntent.SINGLE_STOCK_RECOMMENDATION,
        tickers=["nvda", "NVDA", " msft "],
        horizon=TimeHorizon.SHORT_TERM,
        required_modules=[
            DataModule.PRICES,
            DataModule.PRICES,
            DataModule.FUNDAMENTALS,
        ],
        confidence=0.9,
    )

    assert route.tickers == ["NVDA", "MSFT"]
    assert route.required_modules == [DataModule.PRICES, DataModule.FUNDAMENTALS]
