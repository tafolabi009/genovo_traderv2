# data/macro.py

import pandas as pd
import os
# Using pandas_datareader requires installation: pip install pandas-datareader
# Using NewsAPI requires installation: pip install newsapi-python
try:
    import pandas_datareader.data as web # For FRED
except ImportError:
    print("Warning: pandas_datareader not installed. FRED fetching disabled. `pip install pandas-datareader`")
    web = None
try:
    from newsapi import NewsApiClient # For NewsAPI
except ImportError:
     print("Warning: newsapi-python not installed. News fetching disabled. `pip install newsapi-python`")
     NewsApiClient = None


class MacroDataFetcher:
    """
    Fetches macroeconomic data from FRED and news sentiment using NewsAPI.
    Requires API keys configured in params.yaml.
    """
    def __init__(self, config=None):
        """
        Initializes the MacroDataFetcher.

        Args:
            config (dict, optional): Configuration containing API keys under 'api_keys'.
                                     Example keys: 'fred_api_key', 'newsapi_key'.
                                     Also reads 'macro_config' for series IDs/keywords.
        """
        self.config = config or {}
        self.api_keys = self.config.get('api_keys', {})
        self.macro_config = self.config.get('macro_config', {}) # Specific settings for this module

        self.fred_api_key = self.api_keys.get('fred_api_key', None)
        self.newsapi_key = self.api_keys.get('newsapi_key', None)

        # Set FRED API key environment variable if provided (some versions of pandas_datareader use this)
        if self.fred_api_key:
             os.environ['FRED_API_KEY'] = self.fred_api_key

        # Initialize NewsAPI client if key exists and library is installed
        if self.newsapi_key and NewsApiClient:
            try:
                self.newsapi = NewsApiClient(api_key=self.newsapi_key)
                print("NewsApiClient initialized.")
            except Exception as e:
                print(f"Error initializing NewsApiClient: {e}")
                self.newsapi = None
        else:
            self.newsapi = None
            if not NewsApiClient: print("NewsApiClient not available.")
            elif not self.newsapi_key: print("NewsAPI key not configured.")

        print("MacroDataFetcher initialized.")


    def fetch_fred_data(self, series_ids=None, start_date=None, end_date=None):
        """
        Fetches economic data series from FRED (Federal Reserve Economic Data).

        Args:
            series_ids (list, optional): List of FRED series IDs (e.g., ['DEXUSEU', 'GDP']).
                                         Defaults to config['macro_config']['fred_series_ids'].
            start_date (str or datetime, optional): Start date for data retrieval.
            end_date (str or datetime, optional): End date for data retrieval.

        Returns:
            pd.DataFrame: DataFrame containing the requested FRED data, indexed by date.
                          Returns empty DataFrame if fetching fails.
        """
        if not web:
             print("Error: pandas_datareader not available for FRED fetching.")
             return pd.DataFrame()

        if not series_ids:
            series_ids = self.macro_config.get('fred_series_ids', [])
        if not series_ids:
            print("Warning: No FRED series IDs specified in macro_config.")
            return pd.DataFrame()

        # Note: FRED API key might not be strictly required by pandas_datareader for all series,
        # but it's good practice to have it configured.

        print(f"Fetching FRED data for series: {series_ids}...")
        try:
            # Use pandas_datareader to fetch data
            df_fred = web.DataReader(series_ids, 'fred', start=start_date, end=end_date)
            # Forward fill to handle non-trading days or missing values
            df_fred = df_fred.ffill()
            print(f"Successfully fetched and ffilled {len(df_fred.columns)} FRED series.")
            return df_fred
        except Exception as e:
            print(f"Error fetching FRED data: {e}")
            return pd.DataFrame()


    def fetch_news_sentiment(self, keywords=None, sources=None, from_date=None, to_date=None, language='en', page_size=100):
        """
        Fetches news articles using NewsAPI. Sentiment analysis is NOT implemented here.

        Args:
            keywords (list or str, optional): Keywords or query string. Defaults to config.
            sources (list or str, optional): News sources (e.g., 'bloomberg', 'reuters'). Defaults to config.
            from_date (str or datetime, optional): Start date (YYYY-MM-DD).
            to_date (str or datetime, optional): End date (YYYY-MM-DD).
            language (str): Language code (e.g., 'en').
            page_size (int): Number of results per page (max 100 for NewsAPI developer plan).

        Returns:
            pd.DataFrame: DataFrame containing news headlines, timestamps, sources.
                          Returns empty DataFrame if fetching fails.
        """
        if not self.newsapi:
            print("Error: NewsAPI client not available or not configured.")
            return pd.DataFrame()

        query = keywords or self.macro_config.get('news_keywords', 'forex OR economy OR inflation OR central bank')
        source_list = sources or self.macro_config.get('news_sources', ['bloomberg', 'reuters', 'financial-post']) # Example sources

        print(f"Fetching news for keywords: '{query}' from sources: {source_list}...")
        try:
            # Note: NewsAPI free tier has limitations (e.g., past 24h data only for /top-headlines,
            # /everything endpoint might be restricted or require paid plan for full history)
            # Using 'get_everything' is generally better for historical searches if plan allows.
            all_articles = self.newsapi.get_everything(q=query,
                                                       sources=','.join(source_list) if source_list else None,
                                                       from_param=from_date, # Expects YYYY-MM-DD
                                                       to=to_date,       # Expects YYYY-MM-DD
                                                       language=language,
                                                       sort_by='publishedAt', # 'relevancy', 'popularity', 'publishedAt'
                                                       page_size=page_size)

            articles_list = []
            if all_articles['status'] == 'ok':
                for article in all_articles['articles']:
                    articles_list.append({
                        'timestamp': pd.to_datetime(article['publishedAt']),
                        'source': article['source']['name'],
                        'headline': article['title'],
                        'description': article.get('description', '') # Description might be None
                        # --- Sentiment Analysis Placeholder ---
                        # You would integrate a sentiment library here (VADER, TextBlob, etc.)
                        # 'sentiment_score': calculate_sentiment(article['title'] + ' ' + article.get('description',''))
                    })
            else:
                 print(f"NewsAPI request failed: {all_articles.get('message', 'Unknown error')}")
                 return pd.DataFrame()


            if not articles_list:
                print("No news articles found for the query.")
                return pd.DataFrame()

            df_news = pd.DataFrame(articles_list)
            # Set timestamp as index AFTER creation if needed
            df_news = df_news.set_index('timestamp').sort_index()
            print(f"Successfully fetched {len(df_news)} news articles.")

            # Placeholder: If sentiment was calculated, you might resample/aggregate it here
            # df_sentiment = df_news.resample('H')['sentiment_score'].mean().ffill()
            # return df_sentiment

            return df_news # Return raw articles for now

        except Exception as e:
            print(f"Error fetching or processing NewsAPI data: {e}")
            # Check for specific NewsAPI errors if needed
            # if 'maximum results reached' in str(e).lower(): print("Hit NewsAPI free tier limit?")
            return pd.DataFrame()

    def get_combined_macro_features(self, start_date=None, end_date=None, target_index=None):
        """
        Fetches FRED and News data and attempts to combine them.
        Requires careful handling of different frequencies and alignment.

        Args:
            start_date (str or datetime, optional): Start date for data retrieval.
            end_date (str or datetime, optional): End date for data retrieval.
            target_index (pd.DatetimeIndex, optional): If provided, macro data will be
                                                      reindexed and forward-filled
                                                      to match this index (e.g., trading data index).

        Returns:
            pd.DataFrame: Combined DataFrame, potentially reindexed and forward-filled.
        """
        print("Combining Macro Features...")
        # Fetch data within the date range
        df_fred = self.fetch_fred_data(start_date=start_date, end_date=end_date)
        # NewsAPI date format usually YYYY-MM-DD
        news_start = pd.to_datetime(start_date).strftime('%Y-%m-%d') if start_date else None
        news_end = pd.to_datetime(end_date).strftime('%Y-%m-%d') if end_date else None
        df_news = self.fetch_news_sentiment(from_date=news_start, to_date=news_end) # Fetch raw news articles

        # --- Combination Logic ---
        # This is complex because FRED is usually daily, news is timestamped.
        # Simplest approach: Forward fill FRED data onto a target index.
        # News sentiment would need aggregation (e.g., daily/hourly avg sentiment) before merging.

        # For now, just return FRED data, potentially reindexed
        if target_index is not None:
            if not df_fred.empty:
                print("Reindexing FRED data to target index...")
                # Ensure target_index is timezone-naive or matches df_fred timezone if any
                # df_fred.index = df_fred.index.tz_localize(None) # Example if FRED index is naive
                try:
                    combined_df = df_fred.reindex(target_index, method='ffill')
                    print("FRED data reindexed and forward-filled.")
                    # TODO: Integrate aggregated news sentiment here if implemented
                    return combined_df
                except Exception as e:
                    print(f"Error reindexing FRED data: {e}")
                    return pd.DataFrame() # Return empty on error
            else:
                 print("No FRED data to combine.")
                 return pd.DataFrame() # Return empty frame if no FRED data
        else:
             # Return raw FRED data if no target index provided
             print("Returning raw FRED data (no target index provided for alignment).")
             return df_fred


# --- Factory Function ---
def create_macro_fetcher(config=None):
    """
    Factory function to create the MacroDataFetcher.
    """
    # Pass the main config, the constructor handles nested dicts
    return MacroDataFetcher(config=config)

# Example Usage
if __name__ == '__main__':
    # Load config from file to get API keys etc.
    try:
        import yaml
        with open('../configs/params.yaml', 'r') as f: # Adjust path if needed
             config = yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load params.yaml for example config. {e}")
        # Use a dummy config if loading fails
        config = {
            'api_keys': {'fred_api_key': None, 'newsapi_key': None},
            'macro_config': {
                 'fred_series_ids': ['DEXUSEU'],
                 'news_keywords': 'EURUSD OR ECB OR Fed',
                 'news_sources': ['reuters']
            }
        }

    fetcher = create_macro_fetcher(config)

    print("\n--- Fetching FRED Data Example ---")
    fred_data = fetcher.fetch_fred_data(start_date='2023-01-01', end_date='2023-12-31')
    if not fred_data.empty:
        print(fred_data.tail())

    print("\n--- Fetching News Example ---")
    # NewsAPI free tier might limit date range significantly for 'everything' endpoint
    news_data = fetcher.fetch_news_sentiment(from_date='2024-04-20', to_date='2024-04-27')
    if not news_data.empty:
        print(news_data.head())

    print("\n--- Combined Macro Features Example (No Target Index) ---")
    combined = fetcher.get_combined_macro_features(start_date='2023-01-01', end_date='2023-12-31')
    if not combined.empty:
         print(combined.tail())

