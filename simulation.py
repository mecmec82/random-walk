import streamlit as st
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import time # Import time for potential rate limiting
import math # For ceiling


# --- Streamlit App Configuration ---
st.set_page_config(page_title="Crypto Random Walk Simulation", layout="wide")

st.title("Crypto Price Random Walk Simulation")
st.write("Simulate multiple future crypto price movements using random walks and visualize the median and standard deviation.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

exchange_id = st.sidebar.selectbox("Select Exchange", ['binance', 'coinbase'], index=0) # Default to coinbasepro
st.sidebar.write(f"Using exchange: **{exchange_id}**")


ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days_requested = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=100, value=900, step=100)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=200, step=10) # Increased default for better stats

st.sidebar.write("Data fetched using CCXT. Public data access typically does not require an API key.")
st.sidebar.write("Free data sources may have rate limits or data availability issues.")
st.sidebar.write("Note: `coinbasepro` is generally better supported for trading data than `coinbase`.")


# --- Helper function to fetch historical data with multiple calls ---
# @st.cache_data # Keep caching for efficiency, but be aware changing code inside means cache is cleared
def fetch_historical_ohlcv(exchange_id, ticker, timeframe, num_days, default_limit_per_call=1000):
    """
    Fetches historical OHLCV data by making multiple API calls if needed,
    working backwards from the present using the oldest timestamp.
    Returns a pandas DataFrame with datetime index.
    """
    all_ohlcv = []
    # Use a list to keep track of the oldest timestamps fetched so we can detect if we're stuck
    oldest_timestamps_fetched = []

    # Start fetching from the current time (None means 'now' for the first call)
    # This will be the `since` argument for the fetch_ohlcv call
    current_since_arg = None # timestamp in milliseconds of the *start* of the desired next chunk

    fetched_count = 0
    attempt = 0

    # Adjust limit_per_call based on known exchange limitations if necessary
    # This is a manual adjustment based on observation/documentation
    limit_for_this_exchange = default_limit_per_call
    if exchange_id == 'coinbase':
        # Coinbase (older API) seems to have a smaller limit per call, maybe around 300 or 500
        # Let's try a safer limit here. Note: Even with a smaller limit, pagination might still be broken for this specific driver.
        limit_for_this_exchange = 300 # Or 500? Let's try 300 as it's a common observed limit.
        st.info(f"Adjusting limit_per_call to {limit_for_this_exchange} for {exchange_id} based on potential API limitations.")


    # Estimate max attempts needed *if* pagination works, plus a buffer
    max_fetch_attempts = math.ceil(num_days / limit_for_this_exchange) * 2 + 5 if limit_for_this_exchange > 0 else 10

    st.info(f"Attempting to fetch approximately {num_days} daily candles for {ticker} from {exchange_id} using {limit_for_this_exchange} candles per call.")

    exchange = None
    try:
        exchange = getattr(ccxt, exchange_id)()
        # Optional: set timeout and rateLimit if needed
        # exchange.timeout = 10000 # 10 seconds
        # exchange.rateLimit = 1000 # 1 second between requests - CCXT uses this for implicit sleeps

        # Check if the exchange supports fetchOHLCV
        if not exchange.has or not exchange.has.get('fetchOHLCV'):
             st.error(f"Exchange '{exchange_id}' does not support fetching OHLCV data (fetchOHLCV).")
             return pd.DataFrame() # Return empty DataFrame on failure

        # Load markets and check ticker/timeframe
        exchange.load_markets()
        if ticker not in exchange.markets:
             st.error(f"Trading pair '{ticker}' not found on exchange '{exchange_id}'.")
             return pd.DataFrame()
        if timeframe not in exchange.timeframes:
            st.error(f"Timeframe '{timeframe}' is not supported by exchange '{exchange_id}' for ticker '{ticker}'.")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error initializing exchange or checking ticker/timeframe: {e}")
        return pd.DataFrame()

    # --- Fetch Loop ---
    # We loop until we have enough candles or the exchange indicates no more history
    while fetched_count < num_days and attempt < max_fetch_attempts:
        attempt += 1
        try:
            # fetch_ohlcv(symbol, timeframe, since=timestamp, limit=N)
            # 'since' is the starting timestamp (inclusive)
            # To fetch backwards, we need to determine the `since` timestamp for the *next* oldest chunk.
            # The very first call `since=None` gets the latest data.
            # For subsequent calls, `since` should be derived from the *oldest* candle timestamp of the *previously fetched* chunk minus 1ms.

            st.info(f"Attempt {attempt}: Requesting {limit_for_this_exchange} candles starting from {'None (latest)' if current_since_arg is None else datetime.utcfromtimestamp(current_since_arg / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')}")

            # Fetch data
            chunk = exchange.fetch_ohlcv(ticker, timeframe, since=current_since_arg, limit=limit_for_this_exchange)

            if not chunk:
                st.info(f"Attempt {attempt}: No data returned for the given 'since' timestamp ({'None' if current_since_arg is None else datetime.utcfromtimestamp(current_since_arg / 1000).strftime('%Y-%m-%d %H:%M:%S UTC')}). Reached the end of available historical data or encountered an API issue returning empty chunks.")
                break # No more data

            # --- DETAILED LOGGING AND DUPLICATE CHECK FOR THIS CHUNK ---
            chunk_len = len(chunk)
            oldest_ts_in_chunk = chunk[0][0] if chunk_len > 0 else None
            newest_ts_in_chunk = chunk[-1][0] if chunk_len > 0 else None

            if oldest_ts_in_chunk is not None and newest_ts_in_chunk is not None:
                oldest_date = datetime.utcfromtimestamp(oldest_ts_in_chunk / 1000).strftime('%Y-%m-%d')
                newest_date = datetime.utcfromtimestamp(newest_ts_in_chunk / 1000).strftime('%Y-%m-%d')
                st.info(f"  Attempt {attempt}: Fetched {chunk_len} candles. Date Range: {oldest_date} to {newest_date}.")

                # Add the oldest timestamp of this chunk to our history check list
                oldest_timestamps_fetched.append(oldest_ts_in_chunk)

                # Check if this oldest timestamp is the same as the last one fetched (implies stuck)
                if len(oldest_timestamps_fetched) >= 2 and oldest_timestamps_fetched[-1] >= oldest_timestamps_fetched[-2]:
                     # It should strictly decrease when fetching backwards
                     st.warning(f"  Attempt {attempt}: Oldest timestamp ({oldest_date}) is NOT older than the previous oldest timestamp ({datetime.utcfromtimestamp(oldest_timestamps_fetched[-2] / 1000).strftime('%Y-%m-%d')}). API is likely not paginating correctly. Stopping fetch.")
                     break # Stop if we detect we are stuck

            else:
                 st.info(f"  Attempt {attempt}: Fetched 0 candles.")


            if chunk_len == 0:
                 st.info("Chunk length is 0, stopping fetch.")
                 break # Explicitly break if chunk is empty


            # Append the chunk to our list
            all_ohlcv.extend(chunk)

            # Update the 'since' for the *next* call. This should be the timestamp of the *oldest* candle
            # in the *current* chunk, minus 1 millisecond.
            current_since_arg = oldest_ts_in_chunk - 1


            # Check total fetched count based on the list length
            fetched_count = len(all_ohlcv)

            # Check if we have enough data. If so, we can break early.
            if fetched_count >= num_days:
                 st.info(f"Fetched enough data ({fetched_count} candles >= {num_days}). Stopping fetch loop.")
                 break

            # Implement a small delay to avoid hitting rate limits
            rate_limit_ms = exchange.rateLimit if hasattr(exchange, 'rateLimit') else 1000 # Default 1 second
            sleep_duration = rate_limit_ms / 1000 + 0.1 # Add 100ms buffer
            st.info(f"  Attempt {attempt}: Sleeping for {sleep_duration:.2f} seconds...")
            time.sleep(sleep_duration)


        except ccxt.RateLimitExceeded as e:
            st.warning(f"Attempt {attempt}: Rate limit exceeded: {e}. Waiting 5 seconds and retrying...")
            time.sleep(5) # Wait a fixed amount on rate limit error
            attempt -= 1 # Decrement attempt counter so rate limit retries don't count towards max_fetch_attempts
            if attempt < -10: # Prevent infinite loop if rate limited constantly after few attempts
                 st.error(f"Too many consecutive rate limit errors. Stopping fetch.")
                 break
        except ccxt.BaseError as e:
            st.error(f"Attempt {attempt}: CCXT Error fetching chunk: {e}. Cannot fetch more data.")
            break # Stop fetching on other CCXT errors
        except Exception as e:
            st.error(f"Attempt {attempt}: An unexpected error occurred during chunk fetching: {e}. Cannot fetch more data.")
            break # Stop fetching on other unexpected errors

    if attempt >= max_fetch_attempts:
         st.warning(f"Maximum fetch attempts ({max_fetch_attempts}) reached. Data may be incomplete ({fetched_count} candles fetched).")
    else:
         st.success(f"Finished fetching historical data. Total attempts made: {attempt}.")


    if not all_ohlcv:
        st.warning("No OHLCV data was successfully fetched.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # --- Timestamp Conversion ---
    # CCXT timestamps are specified to be in milliseconds
    try:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    except Exception as e:
        st.error(f"Error converting timestamps to datetime after fetching: {e}. Data format might be unexpected.")
        return pd.DataFrame()

    df = df.dropna(subset=['timestamp_dt'])
    if df.empty:
         st.error("No valid date/timestamp rows remain after conversion and dropping NaT.")
         return pd.DataFrame()

    df.set_index('timestamp_dt', inplace=True)

    # Sort by index (timestamp) and remove duplicates
    # Sorting is crucial because fetching chunks backwards can result in unsorted data,
    # and duplicates might occur at chunk boundaries.
    df = df.sort_index()
    original_len = len(df)
    df = df.loc[~df.index.duplicated(keep='first')] # Keep the first occurrence of any duplicate timestamp
    if len(df) < original_len:
         st.info(f"Removed {original_len - len(df)} duplicate timestamps.")


    st.info(f"Total unique daily candles after sorting and deduplication: {len(df)}")

    # Select the 'close' price and take the last 'num_days' requested
    # We need to take the LAST N days *after* fetching everything and sorting
    historical_data_close = df['close'].tail(num_days)

    if len(historical_data_close) < num_days:
        st.warning(f"Only {len(historical_data_close)} daily candles available for analysis after filtering, sorting, and selecting the last {num_days} days.")

    return historical_data_close


# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):

    # --- Fetch Historical Data ---
    historical_data_close_analyzed = fetch_historical_ohlcv(
        exchange_id=exchange_id,
        ticker=ticker,
        timeframe='1d', # Daily timeframe
        num_days=historical_days_requested,
        # Let the function decide the limit per call based on exchange, or pass a specific one:
        # limit_per_call=500 # Example: Try setting a specific limit here if needed
    )

    # Ensure we have enough data AFTER the fetch attempts
    if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
         st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available for analysis after fetching. Need at least 2 days with valid prices.")
         st.stop()


    # --- Calculate Historical Returns and Volatility ---
    with st.spinner("Calculating historical statistics..."):
        # Calculate log returns from the fetched historical data
        log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

        if len(log_returns) < 1:
            st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility. Need at least 2 consecutive valid prices.")
            st.stop()

        # Calculate mean and standard deviation of log returns
        mean_daily_log_return = log_returns.mean()
        daily_log_volatility = log_returns.std()

        st.subheader("Historical Analysis Results")
        st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}** from **{exchange_id}**:")
        st.info(f"Calculated Mean Daily Log Return: `{mean_daily_log_return:.6f}`")
        st.info(f"Calculated Daily Log Volatility: `{daily_log_volatility:.6f}`")


    # --- Prepare Dates for Plotting (Historical + Simulated) ---
    historical_dates = historical_data_close_analyzed.index
    last_historical_date = historical_dates.max()

    # Generate future dates for the simulation
    simulated_dates = pd.DatetimeIndex([]) # Initialize as empty
    sim_path_length = 0 # Initialize simulation path length

    try:
        # Add 1 because the simulation path includes the starting point (last historical date)
        # but pd.date_range with freq='B' starts *after* the given date
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:]

        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length will be adjusted.")

        sim_path_length = len(simulated_dates) + 1 # Length of the path (start + future days)


    except Exception as date_range_error:
         st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
         simulated_dates = pd.DatetimeIndex([]) # Ensure it's empty on error
         sim_path_length = 0 # Ensure length is 0 on error


    # Combine historical last date with simulated dates for plotting the simulation paths
    plot_sim_dates = pd.DatetimeIndex([]) # Initialize as empty
    if len(simulated_dates) > 0 and sim_path_length > 0:
        last_historical_date_index = pd.DatetimeIndex([last_historical_date])
        plot_sim_dates = last_historical_date_index.append(simulated_dates)
        # Ensure it's a DatetimeIndex after append
        plot_sim_dates = pd.DatetimeIndex(plot_sim_dates)
        # Check final length consistency
        if len(plot_sim_dates) != sim_path_length:
             st.error(f"Mismatch between calculated path length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates)}). Cannot plot.")
             plot_sim_dates = pd.DatetimeIndex([]) # Clear dates if mismatch

    else:
        st.warning("Skipping simulation date axis generation.")


    # --- Run Multiple Simulations ---
    all_simulated_paths = []
    # Only proceed if we have enough historical data and successfully generated simulation dates
    if sim_path_length > 0 and len(historical_data_close_analyzed) > 0:
        with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):
            start_price = historical_data_close_analyzed.iloc[-1]
            for _ in range(num_simulations):
                 # Generate random daily log returns for this simulation path
                # Size matches the number of *steps* into the future (simulation_days)
                simulated_log_returns = np.random.normal(
                    loc=mean_daily_log_return,
                    scale=daily_log_volatility,
                    size=simulation_days
                )

                # --- Calculate Simulated Price Path ---
                simulated_price_path = np.zeros(sim_path_length) # Use the potentially trimmed length
                if sim_path_length > 0:
                    simulated_price_path[0] = start_price

                    # Apply returns up to the number of generated dates/steps
                    for j in range(1, sim_path_length):
                         if j - 1 < len(simulated_log_returns):
                              simulated_price_path[j] = simulated_price_path[j-1] * np.exp(simulated_log_returns[j-1])
                         else:
                              simulated_price_path[j] = simulated_price_path[j-1] # Use last price if somehow runs out of returns


                all_simulated_paths.append(simulated_price_path)
    else:
        st.warning("Skipping simulations as future dates could not be generated or historical data is missing.")

    # --- Calculate Median, Mean, Standard Deviation, and Final Prices ---
    median_prices = np.array([])
    mean_prices = np.array([])
    std_dev_prices = np.array([])
    upper_band = np.array([])
    lower_band = np.array([])
    final_prices = [] # List to store the final price of each valid simulation path

    # Check if simulation paths were generated and have expected length
    if len(all_simulated_paths) > 0 and sim_path_length > 0 and all(len(path) == sim_path_length for path in all_simulated_paths):
        try:
            all_simulated_paths_np = np.vstack(all_simulated_paths) # Stack rows vertically
            prices_at_each_step = all_simulated_paths_np.T # Transpose

            median_prices = np.median(prices_at_each_step, axis=1)
            mean_prices = np.mean(prices_at_each_step, axis=1)
            std_dev_prices = np.std(prices_at_each_step, axis=1)

            # Calculate +/- 1 standard deviation band
            upper_band = mean_prices + std_dev_prices
            lower_band = mean_prices - std_dev_prices

            # Extract final prices for overview summary
            final_prices = [path[-1] for path in all_simulated_paths]


        except Exception as e:
            st.error(f"Error calculating aggregate statistics: {e}")
            # Keep aggregates as empty arrays if calculation fails


    elif len(all_simulated_paths) > 0:
        st.warning("Simulation paths have inconsistent or zero lengths. Cannot calculate aggregate statistics.")
    else:
        st.warning("No simulation paths were generated successfully.")


    # --- Plotting ---
    st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Historical Data
    if not historical_data_close_analyzed.empty:
        ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    # Ensure all necessary arrays for plotting aggregates have the correct length
    if len(plot_sim_dates) > 0 and len(median_prices) == len(plot_sim_dates) and len(upper_band) == len(plot_sim_dates) and len(lower_band) == len(plot_sim_dates):
        # Plot Median Line
        ax.plot(plot_sim_dates, median_prices, label=f'Median Simulated Price ({num_simulations} runs)', color='red', linestyle='-', linewidth=2)

        # Plot Standard Deviation Band
        ax.fill_between(plot_sim_dates, lower_band, upper_band, color='orange', alpha=0.3, label='+/- 1 Std Dev Band')

        # --- Add Labels at the end ---
        if len(plot_sim_dates) > 0:
             final_date = plot_sim_dates[-1]

             ax.text(final_date, median_prices[-1], f" ${median_prices[-1]:.2f}",
                     color='red', fontsize=10, ha='left', va='center', weight='bold')

             ax.text(final_date, upper_band[-1], f" ${upper_band[-1]:.2f}",
                     color='darkorange', fontsize=9, ha='left', va='bottom')

             ax.text(final_date, lower_band[-1], f" ${lower_band[-1]:.2f}",
                     color='darkorange', fontsize=9, ha='left', va='top')

        ax.legend()

    elif len(all_simulated_paths) > 0:
        st.warning("Could not plot simulation aggregates due to data length issues or calculation errors.")
    else:
        st.warning("No simulation data available to plot.")


    ax.set_title(f'{ticker} Price: Historical Data ({exchange_id}) and Random Walk Simulation Aggregates')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)

    try:
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating plot: {e}")

    plt.close(fig)

    # --- Display Final Results ---
    st.subheader("Simulation Results Overview")
    if len(historical_data_close_analyzed) > 0:
        st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${historical_data_close_analyzed.iloc[-1]:.2f}**")

    if len(median_prices) > 0:
         st.write(f"Ran **{num_simulations}** simulations.")
         if len(simulated_dates) > 0 and len(median_prices) > 0:
              st.write(f"Simulated Ending Prices (after {len(simulated_dates)} steps, {simulated_dates[-1].strftime('%Y-%m-%d')}):")
              st.write(f"- Median: **${median_prices[-1]:.2f}**")
              st.write(f"- Mean: ${mean_prices[-1]:.2f}")
              st.write(f"- Std Dev: ${std_dev_prices[-1]:.2f}")
              st.write(f"- +/- 1 Std Dev Range: [${lower_band[-1]:.2f}, ${upper_band[-1]:.2f}]")

              if final_prices:
                  st.write(f"- Actual Min Ending Price: ${np.min(final_prices):.2f}")
                  st.write(f"- Actual Max Ending Price: ${np.max(final_prices):.2f}")
              else:
                   st.warning("Could not calculate min/max from simulated paths.")

         else:
              st.warning("Simulated dates or median prices were not generated successfully.")

    elif len(all_simulated_paths) > 0:
        st.warning("Simulation paths were generated, but aggregation failed.")
    else:
         st.warning("No simulation results to display.")
