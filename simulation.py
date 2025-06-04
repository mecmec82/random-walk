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

exchange_id = st.sidebar.selectbox("Select Exchange", ['coinbase', 'coinbasepro'], index=0) # Added exchange selection
st.sidebar.write(f"Using exchange: **{exchange_id}**")


ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=50, value=300, step=10)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=100, step=10) # Increased default for better stats

st.sidebar.write("Data fetched using CCXT. Public data access typically does not require an API key.")
st.sidebar.write("Free data sources may have rate limits or data availability issues.")
st.sidebar.write("Note: `coinbasepro` is often better supported for trading data than `coinbase`.")


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
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=2000) # Increased limit slightly

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
    simulated_dates = pd.DatetimeIndex([])
    try:
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:]

        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length will be adjusted.")
             sim_path_length = len(simulated_dates) + 1
        else:
             sim_path_length = simulation_days + 1

    except Exception as date_range_error:
         st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
         simulated_dates = pd.DatetimeIndex([])
         sim_path_length = 0


    # Combine historical last date with simulated dates for plotting the simulation paths
    plot_sim_dates = pd.DatetimeIndex([])
    if len(simulated_dates) > 0:
        last_historical_date_index = pd.DatetimeIndex([last_historical_date])
        plot_sim_dates = last_historical_date_index.append(simulated_dates)
        plot_sim_dates = pd.DatetimeIndex(plot_sim_dates)
    else:
        st.warning("Skipping simulation date axis generation.")


    # --- Run Multiple Simulations ---
    all_simulated_paths = []
    if sim_path_length > 0 and len(historical_data_close_analyzed) > 0:
        with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):
            start_price = historical_data_close_analyzed.iloc[-1]
            for _ in range(num_simulations):
                simulated_log_returns = np.random.normal(
                    loc=mean_daily_log_return,
                    scale=daily_log_volatility,
                    size=simulation_days
                )

                simulated_price_path = np.zeros(sim_path_length)
                if sim_path_length > 0:
                    simulated_price_path[0] = start_price
                    for j in range(1, sim_path_length):
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

    if len(all_simulated_paths) > 0 and sim_path_length > 0 and all(len(path) == sim_path_length for path in all_simulated_paths):
        try:
            all_simulated_paths_np = np.vstack(all_simulated_paths)
            prices_at_each_step = all_simulated_paths_np.T

            median_prices = np.median(prices_at_each_step, axis=1)
            mean_prices = np.mean(prices_at_each_step, axis=1)
            std_dev_prices = np.std(prices_at_each_step, axis=1)

            upper_band = mean_prices + std_dev_prices
            lower_band = mean_prices - std_dev_prices

        except Exception as e:
            st.error(f"Error calculating aggregate statistics: {e}")

    elif len(all_simulated_paths) > 0:
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
        if len(upper_band) == len(plot_sim_dates) and len(lower_band) == len(plot_sim_dates):
             ax.fill_between(plot_sim_dates, lower_band, upper_band, color='orange', alpha=0.3, label='+/- 1 Std Dev Band')

             # --- Add Labels at the end ---
             final_date = plot_sim_dates[-1]

             # Median label
             ax.text(final_date, median_prices[-1], f" ${median_prices[-1]:.2f}",
                     color='red', fontsize=10, ha='left', va='center', weight='bold')

             # Upper band label
             ax.text(final_date, upper_band[-1], f" ${upper_band[-1]:.2f}",
                     color='darkorange', fontsize=9, ha='left', va='bottom') # va='bottom' places text slightly above the point

             # Lower band label
             ax.text(final_date, lower_band[-1], f" ${lower_band[-1]:.2f}",
                     color='darkorange', fontsize=9, ha='left', va='top') # va='top' places text slightly below the point


        else:
             st.warning("Length mismatch for plotting std dev band.")

        # Add legend if simulation aggregates were plotted
        ax.legend()

    elif len(all_simulated_paths) > 0:
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

    if len(median_prices) > 0:
         st.write(f"Ran **{num_simulations}** simulations.")
         if len(simulated_dates) > 0:
              st.write(f"Simulated Ending Prices (after {len(simulated_dates)} steps, {simulated_dates[-1].strftime('%Y-%m-%d')}):")
              st.write(f"- Median: **${median_prices[-1]:.2f}**")
              st.write(f"- Mean: ${mean_prices[-1]:.2f}")
              st.write(f"- Std Dev: ${std_dev_prices[-1]:.2f}")
              st.write(f"- +/- 1 Std Dev Range: [${lower_band[-1]:.2f}, ${upper_band[-1]:.2f}]")

              if len(all_simulated_paths) > 0 and len(all_simulated_paths[0]) > 0:
                  final_prices = [path[-1] for path in all_simulated_paths if len(path) > 0]
                  if final_prices: # Check if list is not empty
                      st.write(f"- Actual Min Ending Price: ${np.min(final_prices):.2f}")
                      st.write(f"- Actual Max Ending Price: ${np.max(final_prices):.2f}")
              else:
                   st.warning("Could not calculate min/max from simulated paths.")

         else:
              st.warning("Simulated dates were not generated successfully.")

    elif len(all_simulated_paths) > 0: # means aggregation failed
        st.warning("Simulation paths were generated, but aggregation failed.")
    else:
         st.warning("No simulation results to display.")
