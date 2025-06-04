import streamlit as st
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import time # Import time for potential rate limiting


# --- Streamlit App Configuration ---
st.set_page_config(page_title="Crypto Random Walk Simulation", layout="wide")

st.title("Crypto Price Random Walk Simulation")
st.write("Simulate multiple future crypto price movements using random walks and visualize the median and standard deviation.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

exchange_id = st.sidebar.selectbox("Select Exchange", ['coinbase', 'coinbasepro'], index=1) # Default to coinbasepro
st.sidebar.write(f"Using exchange: **{exchange_id}**")


ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days_requested = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=100, value=900, step=100)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=200, step=10) # Increased default for better stats

st.sidebar.write("Data fetched using CCXT. Public data access typically does not require an API key.")
st.sidebar.write("Free data sources may have rate limits or data availability issues.")
st.sidebar.write("Note: `coinbasepro` is often better supported for trading data than `coinbase`.")


# --- Helper function to fetch historical data with multiple calls ---
@st.cache_data # Cache data fetching results to avoid re-fetching on rerun
def fetch_historical_ohlcv(exchange_id, ticker, timeframe, num_days, limit_per_call=1000):
    """
    Fetches historical OHLCV data by making multiple API calls if needed.
    Returns a pandas DataFrame with datetime index.
    """
    ohlcv_list = []
    since_timestamp = None # Start with no 'since' to get the most recent data

    # We need to fetch roughly num_days, but exchanges give data in chunks by limit.
    # Estimate the number of calls needed, but loop until we have enough or no more data
    fetched_count = 0
    max_retries = 5 # Simple retry mechanism
    retries = 0

    st.info(f"Attempting to fetch ~{num_days} daily candles for {ticker} from {exchange_id}...")

    exchange = getattr(ccxt, exchange_id)()
    # Optional: set timeout and rateLimit if needed
    # exchange.timeout = 10000 # 10 seconds
    # exchange.rateLimit = 1000 # 1 second between requests

    # Check if the exchange supports fetchOHLCV
    if not exchange.has or not exchange.has.get('fetchOHLCV'):
         st.error(f"Exchange '{exchange_id}' does not support fetching OHLCV data (fetchOHLCV).")
         return pd.DataFrame() # Return empty DataFrame on failure

    # Load markets and check ticker/timeframe
    try:
        exchange.load_markets()
        if ticker not in exchange.markets:
             st.error(f"Trading pair '{ticker}' not found on exchange '{exchange_id}'.")
             return pd.DataFrame()
        if timeframe not in exchange.timeframes:
            st.error(f"Timeframe '{timeframe}' is not supported by exchange '{exchange_id}' for ticker '{ticker}'.")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading markets or checking ticker/timeframe: {e}")
        return pd.DataFrame()


    # Loop until we have enough data or the API returns less than the limit
    while fetched_count < num_days and retries < max_retries:
        try:
            # Fetch data. If since_timestamp is None, fetches the latest data.
            # If since_timestamp is set, fetches data FROM that timestamp onwards.
            # We actually want data BEFORE the current oldest timestamp, so we fetch
            # starting from the *last* oldest timestamp with a negative limit?
            # No, the standard pattern is to fetch *backwards* from the latest.
            # We fetch 'limit_per_call' ending at `until_timestamp`. If `until_timestamp` is None, it's the current time.
            # CCXT's fetch_ohlcv typically fetches 'limit' candles ending *before or at* 'since' if called with `since`
            # OR starting *from or at* `since` if called with `since`.
            # The standard way to go backwards is: fetch latest, get oldest timestamp, fetch again with `since=oldest_timestamp`, but some exchanges return the same candle.
            # A more robust way for going backwards is often to fetch *until* a timestamp using params, but that's exchange specific.
            # Let's use the simple `since=oldest_timestamp` approach and rely on pandas for sorting and deduplication.

            # Determine the `until` timestamp for this fetch
            until_timestamp = since_timestamp # This will be None for the first fetch

            # Determine the `since` timestamp for this fetch - to get data *before* `until`
            # For fetching backwards, we usually set `until` and leave `since` as None,
            # or use `since` to get data *from* an earlier point.
            # Let's stick to the `since` parameter to get data *from* an earlier point,
            # effectively fetching from `since` to now (or limit points from `since`).
            # We need to adjust `since` *backwards* in time in each iteration.
            # The first call gets the latest data (since=None). The oldest point in this data becomes the `until` for the *next* call.
            # But fetch_ohlcv doesn't have a standard `until` param, only `since`.
            # Let's use a simple loop: fetch N latest, get oldest timestamp T. Fetch N latest ending at T-1ms.
            # This requires exchange-specific `params` like `endTime` or similar, which is complex.

            # Let's revert to the most reliable common pattern: fetch `limit` candles *starting from* `since`.
            # Start `since` as None to get latest.
            # After getting a chunk, set `since` to the timestamp of the *first* candle in the chunk to get the *next oldest* chunk.
            # Collect chunks, then sort and deduplicate.

            chunk = exchange.fetch_ohlcv(ticker, timeframe, since=since_timestamp, limit=limit_per_call)

            if not chunk:
                st.info("No more historical data available from the exchange.")
                break # No more data

            st.info(f"Fetched {len(chunk)} candles (Total collected: {fetched_count + len(chunk)}).")
            ohlcv_list.append(chunk)
            fetched_count += len(chunk)

            # Get the timestamp of the oldest candle in the current chunk
            since_timestamp = chunk[0][0]

            # Implement a small delay to avoid hitting rate limits, especially on free tiers
            time.sleep(exchange.rateLimit / 1000) # Sleep based on exchange rate limit setting

        except ccxt.RateLimitExceeded as e:
            retries += 1
            st.warning(f"Rate limit exceeded: {e}. Waiting and retrying ({retries}/{max_retries})...")
            time.sleep(exchange.rateLimit / 1000 * (2 ** retries)) # Exponential backoff
        except ccxt.BaseError as e:
            st.error(f"Error fetching chunk: {e}. Cannot fetch more data.")
            break # Stop fetching on other CCXT errors
        except Exception as e:
            st.error(f"An unexpected error occurred during chunk fetching: {e}. Cannot fetch more data.")
            break # Stop fetching on other unexpected errors

    if retries == max_retries:
         st.error(f"Max retries ({max_retries}) reached due to rate limits or errors. Data may be incomplete.")


    # Combine all fetched chunks
    all_ohlcv = [item for sublist in ohlcv_list for item in sublist]

    if not all_ohlcv:
        st.warning("No OHLCV data was successfully fetched.")
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # --- Timestamp Conversion ---
    # CCXT timestamps are in milliseconds
    try:
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
    except Exception as e:
        st.error(f"Error converting timestamps to datetime: {e}. Data format might be unexpected.")
        st.stop() # Cannot proceed without valid dates

    df = df.dropna(subset=['timestamp_dt'])
    df.set_index('timestamp_dt', inplace=True)

    # Sort by index (timestamp) and remove duplicates
    df = df.sort_index()
    df = df.loc[~df.index.duplicated(keep='first')] # Keep the first occurrence of any duplicate timestamp

    st.info(f"Total unique daily candles fetched and processed: {len(df)}")

    # Select the 'close' price and take the last 'num_days'
    historical_data_close = df['close'].tail(num_days)

    if len(historical_data_close) < num_days:
        st.warning(f"Only {len(historical_data_close)} daily candles available for analysis after filtering and selecting the last {num_days} days.")

    return historical_data_close


# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):

    # --- Fetch Historical Data ---
    # Use the helper function to fetch potentially large history
    historical_data_close_analyzed = fetch_historical_ohlcv(
        exchange_id=exchange_id,
        ticker=ticker,
        timeframe='1d', # Daily timeframe
        num_days=historical_days_requested,
        limit_per_call=1000 # Max candles per API call
    )

    if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
         st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available for analysis. Need at least 2 days with valid prices.")
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
                    # Index j goes from 1 to sim_path_length - 1
                    # Index j-1 goes from 0 to sim_path_length - 2
                    # The returns array has length simulation_days.
                    # This works correctly even if sim_path_length < simulation_days + 1 (due to date generation issue)
                    # As it only uses returns up to the number of available steps.
                    for j in range(1, sim_path_length):
                         # Make sure we don't go out of bounds of the simulated_log_returns array
                         # This check is theoretically redundant if sim_path_length matches len(simulated_dates)+1
                         # but defensive coding against potential minor off-by-one issues.
                         if j - 1 < len(simulated_log_returns):
                              simulated_price_path[j] = simulated_price_path[j-1] * np.exp(simulated_log_returns[j-1])
                         else:
                              # Should not happen if lengths are consistent, but fill with last valid price if it does
                              simulated_price_path[j] = simulated_price_path[j-1]


                all_simulated_paths.append(simulated_price_path)
    else:
        st.warning("Skipping simulations as future dates could not be generated or historical data is missing.")

    # --- Calculate Median and Standard Deviation Across Simulations ---
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
            # Now transpose to have prices at each step in columns
            prices_at_each_step = all_simulated_paths_np.T # Transpose

            # Calculate median, mean, std dev for each step (column)
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
        # Check if plot_sim_dates has at least one point to label
        if len(plot_sim_dates) > 0:
             final_date = plot_sim_dates[-1]

             # Median label
             ax.text(final_date, median_prices[-1], f" ${median_prices[-1]:.2f}",
                     color='red', fontsize=10, ha='left', va='center', weight='bold')

             # Upper band label
             ax.text(final_date, upper_band[-1], f" ${upper_band[-1]:.2f}",
                     color='darkorange', fontsize=9, ha='left', va='bottom')

             # Lower band label
             ax.text(final_date, lower_band[-1], f" ${lower_band[-1]:.2f}",
                     color='darkorange', fontsize=9, ha='left', va='top')

             # Optional: Add a horizontal line for the historical closing price on the simulation side
             # if len(historical_data_close_analyzed) > 0:
             #      last_hist_price = historical_data_close_analyzed.iloc[-1]
             #      ax.axhline(last_hist_price, color='gray', linestyle=':', linewidth=1, label='Last Historical Close')


        # Add legend if simulation aggregates were plotted
        ax.legend()

    elif len(all_simulated_paths) > 0:
        st.warning("Could not plot simulation aggregates due to data length issues or calculation errors.")
    else:
        st.warning("No simulation data available to plot.")


    ax.set_title(f'{ticker} Price: Historical Data ({exchange_id}) and Random Walk Simulation Aggregates')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)

    # Use Streamlit's plotting function
    try:
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating plot: {e}")
        st.error("This error might still be related to date formatting within Matplotlib. Check console logs.")

    # Close the figure to prevent memory leaks
    plt.close(fig)

    # --- Display Final Results ---
    st.subheader("Simulation Results Overview")
    if len(historical_data_close_analyzed) > 0:
        st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${historical_data_close_analyzed.iloc[-1]:.2f}**")

    if len(median_prices) > 0: # Check if aggregates were calculated
         st.write(f"Ran **{num_simulations}** simulations.")
         if len(simulated_dates) > 0 and len(median_prices) > 0: # Final check for data availability
              st.write(f"Simulated Ending Prices (after {len(simulated_dates)} steps, {simulated_dates[-1].strftime('%Y-%m-%d')}):")
              st.write(f"- Median: **${median_prices[-1]:.2f}**")
              st.write(f"- Mean: ${mean_prices[-1]:.2f}")
              st.write(f"- Std Dev: ${std_dev_prices[-1]:.2f}")
              st.write(f"- +/- 1 Std Dev Range: [${lower_band[-1]:.2f}, ${upper_band[-1]:.2f}]")

              if final_prices: # Check if the list of final prices was populated
                  st.write(f"- Actual Min Ending Price: ${np.min(final_prices):.2f}")
                  st.write(f"- Actual Max Ending Price: ${np.max(final_prices):.2f}")
              else:
                   st.warning("Could not calculate min/max from simulated paths.")

         else:
              st.warning("Simulated dates or median prices were not generated successfully.")

    elif len(all_simulated_paths) > 0: # means aggregation failed
        st.warning("Simulation paths were generated, but aggregation failed.")
    else:
         st.warning("No simulation results to display.")
