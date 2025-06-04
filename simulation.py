import streamlit as st
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Crypto Random Walk Simulation", layout="wide")

st.title("Crypto Price Random Walk Simulation")
st.write("Simulate multiple future crypto price movements using random walks and visualize the median and standard deviation.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

exchange_id = 'coinbase' # Using 'coinbase' as requested
st.sidebar.write(f"Using exchange: **{exchange_id}**")

ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=50, value=300, step=10)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=20, step=1) # Number of simulations

st.sidebar.write("Data fetched using CCXT. Public data access typically does not require an API key.")
st.sidebar.write("Free data sources may have rate limits or data availability issues.")
st.sidebar.write("Note: The `coinbasepro` exchange ID in CCXT is generally better supported for trading data like OHLCV compared to `coinbase`.")


# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):

    exchange = None
    try:
        # Initialize the exchange
        exchange = getattr(ccxt, exchange_id)()
        # Load markets to ensure everything is set up
        exchange.load_markets()
    except AttributeError:
        st.error(f"Exchange '{exchange_id}' not found. Please check the exchange ID.")
        st.stop()
    except Exception as e:
        st.error(f"Could not initialize exchange '{exchange_id}': {e}")
        st.stop()

    # Ensure the exchange supports fetching OHLCV data for the specific ticker
    if not exchange.has or not exchange.has.get('fetchOHLCV'):
         st.error(f"Exchange '{exchange_id}' does not support fetching OHLCV data (fetchOHLCV).")
         st.stop()

    # Check if the ticker exists and supports the timeframe
    if ticker not in exchange.markets:
         st.error(f"Trading pair '{ticker}' not found on exchange '{exchange_id}'.")
         st.stop()

    # Define the timeframe (daily)
    timeframe = '1d'
    if timeframe not in exchange.timeframes:
        st.error(f"Timeframe '{timeframe}' is not supported by exchange '{exchange_id}' for ticker '{ticker}'.")
        st.stop()


    with st.spinner(f"Fetching historical data for {ticker} from {exchange_id}..."):
        try:
            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=1000) # Fetch up to 1000 candles

            if not ohlcv:
                st.error(f"No historical data fetched for {ticker} from {exchange_id}. Check ticker symbol or exchange status. Data might not be available via the public endpoint.")
                st.stop()

            # Convert to pandas DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # --- Timestamp Conversion ---
            df['timestamp_dt'] = pd.NaT
            converted = False

            try:
                 temp_dt = pd.to_datetime(df['timestamp'], unit='ms')
                 if not temp_dt.isnull().any() and temp_dt.min().year >= 1990 and temp_dt.max().year <= datetime.now().year + 2:
                      df['timestamp_dt'] = temp_dt
                      converted = True
            except Exception:
                 pass

            if not converted:
                 try:
                      temp_dt = pd.to_datetime(df['timestamp'], unit='s')
                      if not temp_dt.isnull().any() and temp_dt.min().year >= 1990 and temp_dt.max().year <= datetime.now().year + 2:
                           df['timestamp_dt'] = temp_dt
                           converted = True
                 except Exception:
                      pass

            if not converted:
                 st.error("Failed to convert timestamps to valid dates using 'ms' or 's'. Data format might be unexpected.")
                 st.stop()

            df = df.dropna(subset=['timestamp_dt'])
            df.set_index('timestamp_dt', inplace=True)

            historical_data_close = df['close'].sort_index()

            if len(historical_data_close) < historical_days:
                st.warning(f"Only {len(historical_data_close)} daily candles available from {exchange_id} for {ticker} after date conversion. Using all available data ({len(historical_data_close)} days) for historical analysis.")
                historical_data_close_analyzed = historical_data_close
            else:
                historical_data_close_analyzed = historical_data_close.tail(historical_days)

            if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
                 st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available or parsed for analysis after filtering. Need at least 2 days with valid prices.")
                 st.stop()

        except ccxt.BaseError as e:
             st.error(f"Error fetching data from {exchange_id}: {e}")
             st.error("This could be a network issue, incorrect ticker, or hitting exchange rate limits.")
             st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred during data processing: {e}")
            st.stop()

    # --- Calculate Historical Returns and Volatility ---
    with st.spinner("Calculating historical statistics..."):
        log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

        if len(log_returns) < 1:
            st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after cleaning. Need at least 2 consecutive valid prices.")
            st.stop()

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
    try:
        # Add 1 because the simulation path includes the starting point (last historical date)
        # but pd.date_range starts *after* the given date with freq='B'
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:]

        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length will be adjusted.")
             # Trim simulation path length to match generated dates + start point
             sim_path_length = len(simulated_dates) + 1
        else:
             sim_path_length = simulation_days + 1

    except Exception as date_range_error:
         st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
         simulated_dates = pd.DatetimeIndex([]) # Ensure it's empty on error
         sim_path_length = 0


    # Combine historical last date with simulated dates for plotting the simulation paths
    plot_sim_dates = pd.DatetimeIndex([]) # Initialize as empty
    if len(simulated_dates) > 0:
        last_historical_date_index = pd.DatetimeIndex([last_historical_date])
        plot_sim_dates = last_historical_date_index.append(simulated_dates)
        plot_sim_dates = pd.DatetimeIndex(plot_sim_dates) # Final check
    else:
        st.warning("Skipping simulation date axis generation.")


    # --- Run Multiple Simulations ---
    all_simulated_paths = []
    if sim_path_length > 0 and len(historical_data_close_analyzed) > 0:
        with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):
            start_price = historical_data_close_analyzed.iloc[-1]
            for _ in range(num_simulations):
                 # Generate random daily log returns for this simulation path
                simulated_log_returns = np.random.normal(
                    loc=mean_daily_log_return,
                    scale=daily_log_volatility,
                    size=simulation_days # Always generate for the requested simulation_days
                )

                # --- Calculate Simulated Price Path ---
                simulated_price_path = np.zeros(sim_path_length) # Use the potentially trimmed length
                if sim_path_length > 0:
                    simulated_price_path[0] = start_price

                    # Only apply returns up to the number of generated dates
                    for j in range(1, sim_path_length):
                        # Use the corresponding random return. Index is j-1 because returns array has length simulation_days.
                        # This works even if sim_path_length < simulation_days + 1
                        simulated_price_path[j] = simulated_price_path[j-1] * np.exp(simulated_log_returns[j-1])

                all_simulated_paths.append(simulated_price_path)
    else:
        st.warning("Skipping simulations as future dates could not be generated or historical data is missing.")

    # --- Calculate Median and Standard Deviation Across Simulations ---
    median_prices = np.array([])
    mean_prices = np.array([])
    std_dev_prices = np.array([])
    upper_band = np.array([])
    lower_band = np.array([])

    if len(all_simulated_paths) > 0 and len(all_simulated_paths[0]) > 0:
        # Convert list of paths into a numpy array for easier column-wise operations
        # Check that all paths have the same length before stacking
        if all(len(path) == sim_path_length for path in all_simulated_paths):
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

        else:
             st.error("Simulation paths have inconsistent lengths. Cannot calculate aggregate statistics.")


    # --- Plotting ---
    st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Historical Data
    if not historical_data_close_analyzed.empty:
        ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    if len(plot_sim_dates) > 0 and len(median_prices) == len(plot_sim_dates):
        # Plot Median Line
        ax.plot(plot_sim_dates, median_prices, label=f'Median Simulated Price ({num_simulations} runs)', color='red', linestyle='-', linewidth=2)

        # Plot Standard Deviation Band
        # Use the same x-axis (plot_sim_dates) for the band
        if len(upper_band) == len(plot_sim_dates) and len(lower_band) == len(plot_sim_dates):
             ax.fill_between(plot_sim_dates, lower_band, upper_band, color='orange', alpha=0.3, label='+/- 1 Std Dev Band')
        else:
             st.warning("Length mismatch for plotting std dev band.")

        # Add legend if simulation aggregates were plotted
        ax.legend()

    elif len(all_simulated_paths) > 0: # Paths were generated, but aggregates couldn't be plotted
        st.warning("Could not plot simulation aggregates due to data length issues.")
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
         # The last point in median_prices and the bands corresponds to the end date
         st.write(f"Simulated Ending Prices (after {len(simulated_dates)} steps, {simulated_dates[-1].strftime('%Y-%m-%d')}):")
         st.write(f"- Median: ${median_prices[-1]:.2f}")
         st.write(f"- Mean: ${mean_prices[-1]:.2f}") # Also show mean for comparison
         st.write(f"- Std Dev: ${std_dev_prices[-1]:.2f}")
         st.write(f"- +/- 1 Std Dev Range: [${lower_band[-1]:.2f}, ${upper_band[-1]:.2f}]")

         # Could also calculate min/max from the actual simulated paths for the range
         if len(all_simulated_paths) > 0 and len(all_simulated_paths[0]) > 0:
             final_prices = [path[-1] for path in all_simulated_paths]
             st.write(f"- Actual Min Ending Price: ${np.min(final_prices):.2f}")
             st.write(f"- Actual Max Ending Price: ${np.max(final_prices):.2f}")

    else:
         st.warning("No simulation results to display.")
