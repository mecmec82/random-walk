import streamlit as st
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Crypto Random Walk Simulation", layout="wide")

st.title("Crypto Price Random Walk Simulation")
st.write("Simulate multiple future crypto price movements using random walks based on historical volatility.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

exchange_id = 'coinbase' # Using 'coinbase' as requested
st.sidebar.write(f"Using exchange: **{exchange_id}**")

ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=50, value=300, step=10)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=20, step=1) # New control

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
            df['timestamp_dt'] = pd.NaT # Initialize a new column for datetime objects
            converted = False
            conversion_unit_used = None

            # Try converting as milliseconds
            try:
                 temp_dt = pd.to_datetime(df['timestamp'], unit='ms')
                 if not temp_dt.isnull().any() and temp_dt.min().year >= 1990 and temp_dt.max().year <= datetime.now().year + 2:
                      df['timestamp_dt'] = temp_dt
                      converted = True
                      conversion_unit_used = 'ms'
                 else:
                     #st.warning(f"Conversion with unit='ms' resulted in unusual/invalid dates.") # Suppress this specific warning in final app unless needed for debug
                     pass
            except Exception: # as ts_error_ms: # Suppress error detail unless needed for debug
                 #st.warning(f"Conversion with unit='ms' failed.")
                 pass

            # If not converted, try converting as seconds
            if not converted:
                 try:
                      temp_dt = pd.to_datetime(df['timestamp'], unit='s')
                      if not temp_dt.isnull().any() and temp_dt.min().year >= 1990 and temp_dt.max().year <= datetime.now().year + 2:
                           df['timestamp_dt'] = temp_dt
                           converted = True
                           conversion_unit_used = 's'
                      #else:
                          #st.warning(f"Conversion with unit='s' resulted in unusual/invalid dates.")
                 except Exception: # as ts_error_s: # Suppress error detail
                      #st.warning(f"Conversion with unit='s' failed.")
                      pass

            if not converted:
                 st.error("Failed to convert timestamps to valid dates using 'ms' or 's'. Data format might be unexpected.")
                 st.stop()

            # st.success(f"Successfully converted timestamps using unit='{conversion_unit_used}'.") # Suppress success message unless needed for debug

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
    try:
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:] # [1:] excludes start date
        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length might be affected.")

    except Exception as date_range_error:
         st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot plot simulation.")
         simulated_dates = pd.DatetimeIndex([]) # Set to empty if error


    # Combine historical last date with simulated dates for plotting the simulation paths
    if len(simulated_dates) > 0:
        last_historical_date_index = pd.DatetimeIndex([last_historical_date])
        plot_sim_dates = last_historical_date_index.append(simulated_dates)
        # Ensure it's a DatetimeIndex after append
        plot_sim_dates = pd.DatetimeIndex(plot_sim_dates)
    else:
        plot_sim_dates = pd.DatetimeIndex([]) # Empty if simulation dates generation failed


    # --- Run Multiple Simulations ---
    all_simulated_paths = []
    if len(simulated_dates) > 0:
        with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):
            for i in range(num_simulations):
                 # Generate random daily log returns for this simulation path
                simulated_log_returns = np.random.normal(
                    loc=mean_daily_log_return,
                    scale=daily_log_volatility,
                    size=simulation_days
                )

                # --- Calculate Simulated Price Path ---
                simulated_price_path = np.zeros(simulation_days + 1)
                simulated_price_path[0] = historical_data_close_analyzed.iloc[-1] # Start from the last historical price

                for j in range(1, simulation_days + 1):
                    simulated_price_path[j] = simulated_price_path[j-1] * np.exp(simulated_log_returns[j-1])

                # Trim path if simulated_dates was trimmed
                if len(simulated_price_path) > len(simulated_dates) + 1:
                     simulated_price_path = simulated_price_path[:len(simulated_dates) + 1]

                all_simulated_paths.append(simulated_price_path)
    else:
        st.warning("Skipping simulations as future dates could not be generated.")


    # --- Plotting ---
    st.subheader("Price Chart: Historical Data and Multiple Simulated Paths")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Historical Data
    if not historical_data_close_analyzed.empty:
        ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Simulated Data
    if len(all_simulated_paths) > 0 and len(plot_sim_dates) > 1:
        for i, path in enumerate(all_simulated_paths):
            if len(plot_sim_dates) == len(path):
                 ax.plot(plot_sim_dates, path, color='red', linestyle='--', alpha=0.3, linewidth=1, label='_nolegend_' if i > 0 else f'{num_simulations} Simulated Paths')
            else:
                 st.warning(f"Skipping simulation path {i+1} due to date and price length mismatch ({len(plot_sim_dates)} dates vs {len(path)} prices).")

        # Add labels only if simulations were attempted and plotted
        if any(len(path) > 1 and len(plot_sim_dates) == len(path) for path in all_simulated_paths):
             # Only add legend if at least one path was successfully plotted
             ax.legend()
        else:
             st.warning("No simulation paths could be plotted due to data length issues.")


    elif len(all_simulated_paths) > 0: # means len(plot_sim_dates) was not > 1
         st.warning("Not enough future dates generated to plot simulations.")
    else:
         st.warning("No simulation paths were generated.")


    ax.set_title(f'{ticker} Price: Historical Data ({exchange_id}) and Random Walk Simulation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')

    ax.grid(True)

    # Use Streamlit's plotting function
    try:
        st.pyplot(fig)
        # st.success("Plot generated successfully.") # Suppress success message
    except Exception as e:
        st.error(f"Error generating plot: {e}")
        st.error("This error might still be related to date formatting within Matplotlib despite efforts to fix. Check console logs.")
        # Optional: print debug info about dates if plot fails again
        # st.write(f"Type of plot_sim_dates: {type(plot_sim_dates)}")
        # if isinstance(plot_sim_dates, pd.DatetimeIndex):
        #      st.write(f"Plot dates range: {plot_sim_dates.min()} to {plot_sim_dates.max()}")
        # st.write(f"First 5 plot_sim_dates: {plot_sim_dates.tolist()[:5]}")

    # Close the figure to prevent memory leaks
    plt.close(fig)

    # --- Display Final Prices (Optional - might be too much for many sims) ---
    st.subheader("Simulation Results Overview")
    if len(historical_data_close_analyzed) > 0:
        st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${historical_data_close_analyzed.iloc[-1]:.2f}**")

    if len(all_simulated_paths) > 0 and len(all_simulated_paths[0]) > 0:
         st.write(f"Ran **{len(all_simulated_paths)}** simulations.")
         # Could add summary stats of ending prices here (mean, std dev, min, max)
         final_prices = [path[-1] for path in all_simulated_paths if len(path) > 0]
         if final_prices:
              st.write(f"Simulated Ending Prices (after {len(simulated_dates)} steps):")
              st.write(f"- Mean: ${np.mean(final_prices):.2f}")
              st.write(f"- Min: ${np.min(final_prices):.2f}")
              st.write(f"- Max: ${np.max(final_prices):.2f}")
         else:
              st.warning("No simulation paths were successfully generated or had data points.")
    else:
         st.warning("No simulated results to display.")
