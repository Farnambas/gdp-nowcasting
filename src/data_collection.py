"""
Data collection module for GDP nowcasting.
Fetches data from FRED API, Google Trends, and generates simulated indicators.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

# FRED API
from fredapi import Fred

# Google Trends
from pytrends.request import TrendReq


class FREDDataCollector:
    """Collects GDP and economic data from FRED API."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("FRED_API_KEY")
        if not self.api_key:
            raise ValueError("FRED API key required. Set FRED_API_KEY environment variable.")
        self.fred = Fred(api_key=self.api_key)

    def get_gdp(self, start_date: str = "2010-01-01") -> pd.DataFrame:
        """Fetch quarterly real GDP data."""
        gdp = self.fred.get_series("GDPC1", observation_start=start_date)
        df = pd.DataFrame({"gdp": gdp})
        df.index.name = "date"
        return df

    def get_gdp_growth(self, start_date: str = "2010-01-01") -> pd.DataFrame:
        """Fetch quarterly GDP growth rate."""
        gdp = self.get_gdp(start_date)
        gdp["gdp_growth"] = gdp["gdp"].pct_change() * 100
        return gdp


class GoogleTrendsCollector:
    """Collects Google Trends data for economic indicators."""

    def __init__(self):
        self.pytrends = TrendReq(hl="en-US", tz=360)

    def get_job_search_trends(
        self,
        keywords: list = None,
        start_date: str = "2010-01-01",
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Fetch Google Trends data for job-related searches.
        Returns monthly data that can be aggregated to quarterly.
        """
        if keywords is None:
            keywords = ["job search", "unemployment", "hiring", "job openings", "layoffs"]

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        timeframe = f"{start_date} {end_date}"

        all_data = []

        # Fetch in batches of 5 (pytrends limit)
        for i in range(0, len(keywords), 5):
            batch = keywords[i:i+5]
            try:
                self.pytrends.build_payload(batch, timeframe=timeframe, geo="US")
                data = self.pytrends.interest_over_time()
                if not data.empty:
                    data = data.drop(columns=["isPartial"], errors="ignore")
                    all_data.append(data)
            except Exception as e:
                print(f"Warning: Could not fetch trends for {batch}: {e}")

        if all_data:
            df = pd.concat(all_data, axis=1)
            df = df.loc[:, ~df.columns.duplicated()]
            return df

        return pd.DataFrame()

    def aggregate_to_quarterly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate monthly Google Trends data to quarterly."""
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        quarterly = df.resample("QE").mean()
        return quarterly


class SimulatedDataGenerator:
    """Generates simulated economic indicators for demonstration."""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_port_traffic(
        self,
        start_date: str = "2010-01-01",
        end_date: str = None,
        freq: str = "QE"
    ) -> pd.DataFrame:
        """
        Generate simulated port traffic data (container throughput).
        Correlated with economic activity with some noise.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n = len(dates)

        # Base trend (growing economy)
        trend = np.linspace(100, 150, n)

        # Seasonal component (Q4 typically higher due to holiday shipping)
        seasonal = 10 * np.sin(np.arange(n) * 2 * np.pi / 4 + np.pi)

        # Cyclical component (business cycle)
        cyclical = 15 * np.sin(np.arange(n) * 2 * np.pi / 20)

        # Random noise
        noise = np.random.normal(0, 5, n)

        # Add recession dip (2020 Q2)
        recession_idx = dates.get_indexer([pd.Timestamp("2020-03-31")], method="nearest")
        if len(recession_idx) > 0 and recession_idx[0] >= 0:
            idx = recession_idx[0]
            for i in range(max(0, idx-1), min(n, idx+3)):
                trend[i] -= 20 * np.exp(-0.5 * (i - idx)**2)

        port_traffic = trend + seasonal + cyclical + noise

        df = pd.DataFrame({
            "port_traffic": port_traffic
        }, index=dates)
        df.index.name = "date"

        return df

    def generate_electricity_consumption(
        self,
        start_date: str = "2010-01-01",
        end_date: str = None,
        freq: str = "QE"
    ) -> pd.DataFrame:
        """
        Generate simulated electricity consumption data.
        Industrial electricity use correlates with economic output.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        dates = pd.date_range(start=start_date, end=end_date, freq=freq)
        n = len(dates)

        # Base trend (growing demand)
        trend = np.linspace(1000, 1200, n)

        # Seasonal component (summer AC, winter heating)
        quarters = np.array([d.quarter for d in dates])
        seasonal = np.where(quarters.isin([1, 3]), 50, -30)

        # Economic correlation component
        economic_cycle = 40 * np.sin(np.arange(n) * 2 * np.pi / 16)

        # Random noise
        noise = np.random.normal(0, 20, n)

        # COVID dip
        recession_idx = dates.get_indexer([pd.Timestamp("2020-03-31")], method="nearest")
        if len(recession_idx) > 0 and recession_idx[0] >= 0:
            idx = recession_idx[0]
            for i in range(max(0, idx-1), min(n, idx+3)):
                trend[i] -= 80 * np.exp(-0.5 * (i - idx)**2)

        electricity = trend + seasonal + economic_cycle + noise

        df = pd.DataFrame({
            "electricity_consumption": electricity
        }, index=dates)
        df.index.name = "date"

        return df


def collect_all_data(
    fred_api_key: str,
    start_date: str = "2010-01-01",
    use_simulated_trends: bool = False
) -> pd.DataFrame:
    """
    Collect all data sources and merge into a single DataFrame.

    Args:
        fred_api_key: FRED API key
        start_date: Start date for data collection
        use_simulated_trends: If True, simulate Google Trends data instead of fetching

    Returns:
        DataFrame with all features aligned to quarterly frequency
    """
    print("Collecting GDP data from FRED...")
    fred = FREDDataCollector(api_key=fred_api_key)
    gdp_data = fred.get_gdp_growth(start_date=start_date)

    print("Generating simulated port traffic data...")
    sim = SimulatedDataGenerator()
    port_data = sim.generate_port_traffic(start_date=start_date)

    print("Generating simulated electricity consumption data...")
    electricity_data = sim.generate_electricity_consumption(start_date=start_date)

    if use_simulated_trends:
        print("Generating simulated Google Trends data...")
        dates = gdp_data.index
        n = len(dates)
        np.random.seed(42)
        trends_data = pd.DataFrame({
            "job_search": 50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 8) + np.random.normal(0, 5, n),
            "unemployment": 40 + 25 * np.sin(np.arange(n) * 2 * np.pi / 12 + np.pi) + np.random.normal(0, 5, n),
            "hiring": 55 + 15 * np.sin(np.arange(n) * 2 * np.pi / 10) + np.random.normal(0, 5, n),
        }, index=dates)
    else:
        print("Collecting Google Trends data...")
        try:
            trends = GoogleTrendsCollector()
            trends_monthly = trends.get_job_search_trends(
                keywords=["job search", "unemployment", "hiring"],
                start_date=start_date
            )
            if not trends_monthly.empty:
                trends_data = trends.aggregate_to_quarterly(trends_monthly)
            else:
                print("Warning: Could not fetch Google Trends, using simulated data")
                use_simulated_trends = True
                dates = gdp_data.index
                n = len(dates)
                np.random.seed(42)
                trends_data = pd.DataFrame({
                    "job_search": 50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 8) + np.random.normal(0, 5, n),
                    "unemployment": 40 + 25 * np.sin(np.arange(n) * 2 * np.pi / 12 + np.pi) + np.random.normal(0, 5, n),
                    "hiring": 55 + 15 * np.sin(np.arange(n) * 2 * np.pi / 10) + np.random.normal(0, 5, n),
                }, index=dates)
        except Exception as e:
            print(f"Warning: Google Trends error ({e}), using simulated data")
            dates = gdp_data.index
            n = len(dates)
            np.random.seed(42)
            trends_data = pd.DataFrame({
                "job_search": 50 + 20 * np.sin(np.arange(n) * 2 * np.pi / 8) + np.random.normal(0, 5, n),
                "unemployment": 40 + 25 * np.sin(np.arange(n) * 2 * np.pi / 12 + np.pi) + np.random.normal(0, 5, n),
                "hiring": 55 + 15 * np.sin(np.arange(n) * 2 * np.pi / 10) + np.random.normal(0, 5, n),
            }, index=dates)

    # Merge all data
    print("Merging all data sources...")
    df = gdp_data.copy()

    # Align indices
    df = df.join(port_data, how="left")
    df = df.join(electricity_data, how="left")
    df = df.join(trends_data, how="left")

    # Forward fill any missing values
    df = df.ffill()

    print(f"Collected data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


if __name__ == "__main__":
    # Test data collection
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("FRED_API_KEY")
    if api_key:
        df = collect_all_data(api_key, use_simulated_trends=True)
        print("\nData preview:")
        print(df.head(10))
    else:
        print("Set FRED_API_KEY to test data collection")
