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
            # Fetch more data than requested historical_days to ensure we have enough trading days
            # Use a generous limit. Free tiers might not support large limits or pagination easily.
            # The actual number of returned candles might be less than the limit.
            ohlcv = exchange.fetch_ohlcv(ticker, timeframe, limit=1000) # Fetch up to 1000 candles

            if not ohlcv:
                st.error(f"No historical data fetched for {ticker} from {exchange_id}. Check ticker symbol or exchange status. Data might not be available via the public endpoint.")
                st.stop()

            # Convert to pandas DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # --- IMPROVED TIMESTAMP CONVERSION ---
            # Inspect first few raw timestamps to guess the unit
            st.info(f"First 5 raw timestamps: {[ts for ts in df['timestamp'].head().tolist()]}")

            # Attempt conversion, trying milliseconds first, then seconds
            converted = False
            for unit in ['ms', 's']:
                 try:
                      df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit=unit)
                      # Check if dates are reasonable (e.g., after 1990 and not too far in future)
                      if df['timestamp_dt'].min().year >= 1990 and df['timestamp_dt'].max().year <= datetime.now().year + 2:
                           st.success(f"Successfully converted timestamps using unit='{unit}'. Dates look reasonable.")
                           df['timestamp'] = df['timestamp_dt'] # Overwrite the original timestamp column with datetime objects
                           converted = True
                           break # Found a working unit, exit loop
                      else:
                          # Dates are outside expected range, this unit probably isn't right
                          st.warning(f"Conversion with unit='{unit}' resulted in unusual dates ({df['timestamp_dt'].min().date()} to {df['timestamp_dt'].max().date()}). Trying next unit.")
                 except Exception as ts_error:
                      st.warning(f"Conversion with unit='{unit}' failed: {ts_error}. Trying next unit.")

            if not converted:
                 st.error("Failed to convert timestamps to valid dates using 'ms' or 's'. Data format might be unexpected.")
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

        # Check if enough data remained after calculating returns
        if len(log_returns) < 1: # Need at least one return to calculate mean/std
            st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after cleaning. Need at least 2 consecutive valid prices.")
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
    last_historical_date = historical_dates.max()

    # Generate future dates for the simulation, starting after the last historical date
    # Crypto markets are 24/7, but using 'B' (business days) aligns better with stock charts visually.
    # Let's handle potential index issues more robustly
    try:
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:]
         # Ensure the length is correct
        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Trimming simulation results.")
             # Trim simulation data if dates couldn't be generated
             simulated_price_path = simulated_price_path[:len(simulated_dates) + 1]
             if len(simulated_price_path) > 0:
                  st.warning(f"Simulated price path trimmed to length {len(simulated_price_path)}.")

    except Exception as date_range_error:
         st.error(f"Error generating future dates: {date_range_error}. Cannot plot simulation.")
         # Set simulated dates and path to empty so plotting is skipped or handled gracefully
         simulated_dates = pd.DatetimeIndex([])
         simulated_price_path = np.array([])


    # --- Plotting ---
    st.subheader("Price Chart: Historical and Simulated")

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Historical Data
    if not historical_data_close_analyzed.empty:
        ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue')
    else:
        st.warning("No historical data available to plot.")


    # Plot Simulated Data
    if len(simulated_dates) > 0 and len(simulated_price_path) > 1: # Need at least 2 points to draw a line
        # The x-axis for the simulated path includes the last historical date + the future simulated dates
        plot_sim_dates = np.concatenate(([last_historical_date], simulated_dates))

        if len(plot_sim_dates) == len(simulated_price_path):
             ax.plot(plot_sim_dates, simulated_price_path, label=f'Simulated Future Price ({len(simulated_dates)} days)', color='red', linestyle='--')
        else:
             st.error("Error plotting simulation: Date and price length mismatch. Skipping simulation plot.")
             st.write(f"Plot dates length: {len(plot_sim_dates)}, Simulated prices length: {len(simulated_price_path)}")
    elif len(simulated_dates) > 0 or len(simulated_price_path) > 0:
         st.warning("Not enough data points generated for simulation plot.")


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
    if len(simulated_dates) > 0 and len(simulated_price_path) > 0: # Check if simulation ran successfully
         st.write(f"**Simulated Price** after {len(simulated_dates)} steps ({simulated_dates[-1].strftime('%Y-%m-%d')}): **${simulated_price_path[-1]:.2f}**")
    else:
         st.warning("No simulated results to display.")
