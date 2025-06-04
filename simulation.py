import streamlit as st
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Crypto Random Walk Simulation Debug", layout="wide")

st.title("Crypto Price Random Walk Simulation (Debugging Dates)")
st.write("Simulate future crypto price movements using a random walk based on historical volatility, using free data via CCXT.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

# Use Coinbase as the exchange ID as requested, but add note
exchange_id = 'coinbase' # <-- Changed from 'coinbasepro'
st.sidebar.write(f"Using exchange: **{exchange_id}**")

# Ticker for Bitcoin vs USD on most exchanges (CCXT format)
# Allow user to input but default to BTC/USD
ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=50, value=300, step=10)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)

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

            # --- IMPROVED TIMESTAMP CONVERSION WITH MORE DEBUGGING ---
            st.info(f"Total fetched OHLCV rows: {len(df)}")
            st.info(f"First 5 raw timestamps from CCXT: {[ts for ts in df['timestamp'].head().tolist()]}")

            # Attempt conversion, trying milliseconds first, then seconds
            df['timestamp_dt'] = pd.NaT # Initialize a new column for datetime objects
            converted = False
            conversion_unit_used = None

            # Try converting as milliseconds
            try:
                 temp_dt = pd.to_datetime(df['timestamp'], unit='ms')
                 # Check if resulting dates are reasonable
                 if not temp_dt.isnull().any() and temp_dt.min().year >= 1990 and temp_dt.max().year <= datetime.now().year + 2:
                      df['timestamp_dt'] = temp_dt
                      converted = True
                      conversion_unit_used = 'ms'
                 else:
                     st.warning(f"Conversion with unit='ms' resulted in unusual/invalid dates ({temp_dt.min()} to {temp_dt.max()}). Trying seconds.")
            except Exception as ts_error_ms:
                 st.warning(f"Conversion with unit='ms' failed: {ts_error_ms}. Trying seconds.")

            # If not converted, try converting as seconds
            if not converted:
                 try:
                      temp_dt = pd.to_datetime(df['timestamp'], unit='s')
                      # Check if resulting dates are reasonable
                      if not temp_dt.isnull().any() and temp_dt.min().year >= 1990 and temp_dt.max().year <= datetime.now().year + 2:
                           df['timestamp_dt'] = temp_dt
                           converted = True
                           conversion_unit_used = 's'
                      else:
                          st.warning(f"Conversion with unit='s' resulted in unusual/invalid dates ({temp_dt.min()} to {temp_dt.max()}).")
                 except Exception as ts_error_s:
                      st.warning(f"Conversion with unit='s' failed: {ts_error_s}.")

            if not converted:
                 st.error("Failed to convert timestamps to valid dates using 'ms' or 's'. Data format might be unexpected.")
                 st.stop()

            st.success(f"Successfully converted timestamps using unit='{conversion_unit_used}'.")

            # Filter out any rows where conversion failed (though hopefully handled by checks)
            df = df.dropna(subset=['timestamp_dt'])
            df.set_index('timestamp_dt', inplace=True) # Set the valid datetime column as index

            # We want the 'close' price and need it sorted by date (ascending)
            historical_data_close = df['close'].sort_index()

            # Ensure we have enough data and take the last 'historical_days' trading days
            if len(historical_data_close) < historical_days:
                st.warning(f"Only {len(historical_data_close)} daily candles available from {exchange_id} for {ticker} after date conversion. Using all available data ({len(historical_data_close)} days) for historical analysis.")
                historical_data_close_analyzed = historical_data_close
            else:
                historical_data_close_analyzed = historical_data_close.tail(historical_days)

            if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
                 st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available or parsed for analysis after filtering. Need at least 2 days with valid prices.")
                 st.stop()

            # --- More Debugging Info on Final Historical Data ---
            st.info(f"Historical data shape for analysis: {historical_data_close_analyzed.shape}")
            st.info(f"Historical data index type: {type(historical_data_close_analyzed.index)}")
            if not historical_data_close_analyzed.empty:
                 st.info(f"Historical date range: {historical_data_close_analyzed.index.min().strftime('%Y-%m-%d')} to {historical_data_close_analyzed.index.max().strftime('%Y-%m-%d')}")


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

    # --- Plotting Historical Data Separately ---
    st.subheader("Plot 1: Historical Price Data")
    if not historical_data_close_analyzed.empty:
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        ax1.plot(historical_data_close_analyzed.index, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue')
        ax1.set_title(f'{ticker} Price: Historical Data ({exchange_id})')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True)
        try:
            st.pyplot(fig1)
            st.success("Historical plot generated successfully.")
        except Exception as e:
            st.error(f"Error generating historical plot: {e}")
            st.error("This is likely the date formatting issue. Check the 'Historical data index type' and 'Historical date range' messages above.")
        plt.close(fig1)
    else:
        st.warning("No historical data available to plot.")


    # --- Prepare and Run Simulation (Only if historical data was successfully processed) ---
    if not historical_data_close_analyzed.empty and len(log_returns) >= 1: # Ensure historical analysis was successful
        with st.spinner(f"Running simulation for {simulation_days} days..."):
            last_price = historical_data_close_analyzed.iloc[-1] # Get the very last historical price

            # Generate random daily log returns for the simulation period
            simulated_log_returns = np.random.normal(
                loc=mean_daily_log_return,
                scale=daily_log_volatility,
                size=simulation_days
            )

            # --- Calculate Simulated Price Path ---
            simulated_price_path = np.zeros(simulation_days + 1)
            simulated_price_path[0] = last_price # The simulation starts at the last real price

            for i in range(1, simulation_days + 1):
                simulated_price_path[i] = simulated_price_path[i-1] * np.exp(simulated_log_returns[i-1])

            # --- Prepare Dates for Plotting Simulation ---
            historical_dates = historical_data_close_analyzed.index
            last_historical_date = historical_dates.max()

            try:
                simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:]
                if len(simulated_dates) != simulation_days:
                     st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Trimming simulation results.")
                     simulated_price_path = simulated_price_path[:len(simulated_dates) + 1]
                     if len(simulated_price_path) > 0:
                          st.warning(f"Simulated price path trimmed to length {len(simulated_price_path)}.")

            except Exception as date_range_error:
                 st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot plot simulation.")
                 simulated_dates = pd.DatetimeIndex([])
                 simulated_price_path = np.array([])

            # --- Plotting Simulated Data Separately ---
            st.subheader("Plot 2: Simulated Future Price")

            if len(simulated_dates) > 0 and len(simulated_price_path) > 1: # Need at least 2 points to draw a line
                fig2, ax2 = plt.subplots(figsize=(14, 7))

                # The x-axis for the simulated path includes the last historical date + the future simulated dates
                plot_sim_dates = np.concatenate(([last_historical_date], simulated_dates))

                if len(plot_sim_dates) == len(simulated_price_path):
                     ax2.plot(plot_sim_dates, simulated_price_path, label=f'Simulated Future Price ({len(simulated_dates)} days)', color='red', linestyle='--')
                     ax2.set_title(f'{ticker} Price: Simulated Future Random Walk')
                     ax2.set_xlabel('Date')
                     ax2.set_ylabel('Price ($)')
                     ax2.legend()
                     ax2.grid(True)
                     try:
                          st.pyplot(fig2)
                          st.success("Simulation plot generated successfully.")
                     except Exception as e:
                          st.error(f"Error generating simulation plot: {e}")
                          st.error("This is likely related to the dates/index used for the simulation plot.")
                     plt.close(fig2)
                else:
                     st.error("Error plotting simulation: Date and price length mismatch. Skipping simulation plot.")
                     st.write(f"Plot dates length: {len(plot_sim_dates)}, Simulated prices length: {len(simulated_price_path)}")
            elif len(simulated_dates) > 0 or len(simulated_price_path) > 0:
                 st.warning("Not enough data points generated for simulation plot.")
            else:
                 st.warning("No simulated data available to plot.")

        # --- Display Final Prices (Only if simulation ran) ---
        st.subheader("Simulation Results")
        if len(historical_data_close_analyzed) > 0:
            st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${historical_data_close_analyzed.iloc[-1]:.2f}**")
        if len(simulated_dates) > 0 and len(simulated_price_path) > 0:
             st.write(f"**Simulated Price** after {len(simulated_dates)} steps ({simulated_dates[-1].strftime('%Y-%m-%d')}): **${simulated_price_path[-1]:.2f}**")
        else:
             st.warning("No simulated results to display.")

    else:
         st.error("Skipping simulation and second plot due to insufficient or invalid historical data.")
