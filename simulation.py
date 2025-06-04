import streamlit as st
import ccxt
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Crypto Random Walk Simulation", layout="wide")

st.title("Crypto Price Random Walk Simulation")
st.write("Simulate future crypto price movements using a random walk based on historical volatility, using free data via CCXT.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

# Use Coinbase as the exchange ID as requested
exchange_id = 'coinbase' # <-- Changed from 'coinbasepro'
st.sidebar.write(f"Using exchange: **{exchange_id}**")

# Ticker for Bitcoin vs USD on most exchanges (CCXT format)
# Allow user to input but default to BTC/USD
ticker = st.sidebar.text_input("Trading Pair (e.g., BTC/USD)", 'BTC/USD').upper()

historical_days = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=50, value=300, step=10)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)

st.sidebar.write("Data fetched using CCXT. Public data access typically does not require an API key.")
st.sidebar.write("Free data sources may have rate limits.")
st.sidebar.write("Note: 'coinbasepro' is often recommended over 'coinbase' for trading data if available.")


# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):

    exchange = None
    try:
        # Initialize the exchange
        exchange = getattr(ccxt, exchange_id)()
        # Set timeout if needed, though not strictly necessary for public data
        # exchange.timeout = 10000 # 10 seconds
    except AttributeError:
        st.error(f"Exchange '{exchange_id}' not found. Please check the exchange ID.")
        st.stop()
    except Exception as e:
        st.error(f"Could not initialize exchange '{exchange_id}': {e}")
        st.stop()

    # Ensure the exchange supports fetching OHLCV data
    # Not all exchanges support fetchOHLCV, especially older APIs like potentially 'coinbase'
    # Although CCXT tries to abstract, let's check dynamically
    if not exchange.has or not exchange.has.get('fetchOHLCV'):
         st.error(f"Exchange '{exchange_id}' does not support fetching OHLCV data (fetchOHLCV).")
         st.stop()


    # Define the timeframe (daily)
    timeframe = '1d'

    with st.spinner(f"Fetching historical data for {ticker} from {exchange_id}..."):
        try:
            # Fetch OHLCV data
            # Fetch more data than requested historical_days to ensure we have enough trading days
            # Use a generous limit. Free tiers might not support large limits or pagination easily.
            # The actual number of returned candles might be less than the limit.
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=1000) # Fetch up to 1000 candles

            if not ohlcv:
                st.error(f"No historical data fetched for {ticker} from {exchange_id}. Check ticker symbol or exchange status. Data might not be available via the public endpoint.")
                st.stop()

            # Convert to pandas DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # --- CRITICAL CHANGE: Try different timestamp unit ---
            # CCXT timestamps are typically milliseconds, but older APIs like 'coinbase' might be seconds.
            # The matplotlib error suggests a date way out of bounds, likely from wrong timestamp unit.
            try:
                 # Assume milliseconds first (standard CCXT)
                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                 # Quick check if resulting dates are reasonable (e.g., not year 1970 or 50000)
                 if df['timestamp'].min().year < 1990 or df['timestamp'].max().year > datetime.now().year + 10:
                     st.warning("Timestamps from 'ms' conversion look unusual. Trying 's' (seconds).")
                     df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                     if df['timestamp'].min().year < 1990 or df['timestamp'].max().year > datetime.now().year + 10:
                          st.error("Timestamp conversion resulted in unreasonable dates even with 's'. Data might be corrupted or in an unexpected format.")
                          st.stop()
            except Exception as ts_error:
                 st.error(f"Error converting timestamp: {ts_error}. Trying 's' (seconds).")
                 try:
                      df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                      if df['timestamp'].min().year < 1990 or df['timestamp'].max().year > datetime.now().year + 10:
                           st.error("Timestamp conversion resulted in unreasonable dates even with 's'. Data might be corrupted or in an unexpected format.")
                           st.stop()
                 except Exception as ts_error_s:
                      st.error(f"Error converting timestamp with 's': {ts_error_s}. Could not parse timestamps correctly.")
                      st.stop()

            df.set_index('timestamp', inplace=True)

            # We want the 'close' price and need it sorted by date (ascending)
            historical_data_close = df['close'].sort_index()

            # Ensure we have enough data and take the last 'historical_days' trading days
            if len(historical_data_close) < historical_days:
                st.warning(f"Only {len(historical_data_close)} daily candles available from {exchange_id} for {ticker}. Using all available data ({len(historical_data_close)} days) for historical analysis.")
                historical_data_close_analyzed = historical_data_close
            else:
                historical_data_close_analyzed = historical_data_close.tail(historical_days)

            if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
                 st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available or parsed for analysis after filtering. Need at least 2 days.")
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

        # Check if enough data remained after calculating returns
        if log_returns.empty:
            st.error("Not enough valid historical data (need at least 2 days with non-zero prices) to calculate returns and volatility.")
            st.stop()

        mean_daily_log_return = log_returns.mean()
        daily_log_volatility = log_returns.std()

        st.subheader("Historical Analysis Results")
        st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}** from **{exchange_id}**:")
        st.info(f"Calculated Mean Daily Log Return: `{mean_daily_log_return:.6f}`")
        st.info(f"Calculated Daily Log Volatility: `{daily_log_volatility:.6f}`")

    # --- Prepare and Run Simulation ---
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

    # --- Prepare Dates for Plotting ---
    historical_dates = historical_data_close_analyzed.index
    last_historical_date = historical_dates.max() # Use max() in case sorting was weird, though index should be sorted

    # Generate future dates for the simulation, starting after the last historical date
    # Crypto markets are 24/7, but using 'B' (business days) aligns better with stock charts visually.
    simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:] # [1:] excludes start date


    # Ensure simulated dates match simulation steps length
    # This can happen if the period spans holidays that pandas doesn't recognize as 'business' days,
    # or if the simulation days span a very large period.
    if len(simulated_dates) != simulation_days:
         st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Trimming simulation results.")
         # Trim simulation data if dates couldn't be generated
         simulated_price_path = simulated_price_path[:len(simulated_dates) + 1]


    # --- Plotting ---
    st.subheader("Price Chart: Historical and Simulated")

    # Use matplotlib.use('Agg') if running in environments without a display, though Streamlit handles this well.
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Historical Data
    ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue')

    # Plot Simulated Data
    # The x-axis for the simulated path includes the last historical date + the future simulated dates
    plot_sim_dates = np.concatenate(([last_historical_date], simulated_dates))

    if len(plot_sim_dates) == len(simulated_price_path):
         ax.plot(plot_sim_dates, simulated_price_path, label=f'Simulated Future Price ({len(simulated_dates)} days)', color='red', linestyle='--')
    else:
         st.error("Error plotting simulation: Date and price length mismatch.")
         st.write(f"Plot dates length: {len(plot_sim_dates)}, Simulated prices length: {len(simulated_price_path)}")


    ax.set_title(f'{ticker} Price: Historical Data ({exchange_id}) and Random Walk Simulation')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.legend()
    ax.grid(True)

    # Use Streamlit's plotting function
    st.pyplot(fig)

    # Close the figure to prevent memory leaks
    plt.close(fig)

    # --- Display Final Prices ---
    st.subheader("Simulation Results")
    if len(historical_data_close_analyzed) > 0:
        st.write(f"**Last Historical Price** ({historical_dates[-1].strftime('%Y-%m-%d')}): **${last_price:.2f}**")
    if len(simulated_dates) > 0:
         st.write(f"**Simulated Price** after {len(simulated_dates)} steps ({simulated_dates[-1].strftime('%Y-%m-%d')}): **${simulated_price_path[-1]:.2f}**")
    else:
         st.warning("No simulated dates or prices were generated.")
        
