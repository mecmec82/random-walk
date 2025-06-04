import streamlit as st
import yfinance as yf # Use yfinance instead of ccxt or alpha vantage
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import time # Import time for potential rate limiting (though less critical with yfinance)
import math # For ceiling

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Market Random Walk Simulation (yfinance)", layout="wide")

st.title("Market Price Random Walk Simulation")
st.write("Simulate multiple future price movements using random walks based on historical volatility, using free data via yfinance.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

# Use a general ticker input that works for both stocks and crypto in yfinance
ticker = st.sidebar.text_input("Ticker Symbol (e.g., SPY, BTC-USD)", 'BTC-USD').upper()

# Allow specifying a range of historical days
historical_days_requested = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=100, value=900, step=100)

simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=200, step=10)

st.sidebar.write("Data fetched using yfinance (Yahoo Finance).")
st.sidebar.write("Note: Yahoo Finance data may have occasional inaccuracies or downtime.")


# --- Helper function to fetch historical data using yfinance ---
@st.cache_data # Cache data fetching results to avoid re-fetching on rerun
def fetch_historical_data(ticker, num_days):
    """
    Fetches historical data for the last num_days using yfinance.
    Returns a pandas Series of closing prices with datetime index.
    """
    st.info(f"Fetching historical data for {ticker}...")

    # Determine start date: go back enough calendar days to cover num_days *trading* days
    # Using 1.5x as a conservative estimate for stocks (approx 252 trading days/year)
    # For crypto (7 days/week), 1x is enough, but 1.5x is safe for both.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days * 1.5)

    try:
        # yf.download returns a DataFrame
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series() # Return empty Series

        # We need the 'Close' price and ensure it's sorted by date (ascending)
        # yfinance usually returns data sorted ascending, but explicit sort is safe
        historical_data_close = data['Close'].sort_index()

        # Take the last 'num_days' trading days
        historical_data_close_analyzed = historical_data_close.tail(num_days)

        if len(historical_data_close_analyzed) < num_days:
             st.warning(f"Only {len(historical_data_close_analyzed)} trading days available for analysis after fetching and selecting the last {num_days} days.")
        else:
             st.success(f"Successfully fetched {len(historical_data_close_analyzed)} trading days.")


        return historical_data_close_analyzed

    except Exception as e:
        st.error(f"Error fetching data for {ticker} using yfinance: {e}")
        st.error("Please check the ticker symbol and your internet connection.")
        return pd.Series() # Return empty Series on error


# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):

    # --- Fetch Historical Data ---
    historical_data_close_analyzed = fetch_historical_data(
        ticker=ticker,
        num_days=historical_days_requested
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
        st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}**:")
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
        # pd.date_range starts *after* the given start date with freq='B'
        # For crypto (7 days/week), 'D' might be more appropriate, but 'B' looks visually consistent.
        # Let's stick with 'B' for general market simulation feel.
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
        # Ensure last_historical_date is a DatetimeIndex before appending
        last_historical_date_index = pd.DatetimeIndex([last_historical_date])
        plot_sim_dates = last_historical_date_index.append(simulated_dates)
        # Ensure it's a DatetimeIndex after append (belt and suspenders)
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
                    # Index j goes from 1 to sim_path_length - 1
                    # Index j-1 goes from 0 to sim_path_length - 2
                    # The returns array has length simulation_days.
                    # This works correctly even if sim_path_length < simulation_days + 1 (due to date generation issue)
                    # As it only uses returns up to the number of available steps.
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
        # Check if plot_sim_dates has at least one point to label (should be > 0 if we are here)
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


    ax.set_title(f'{ticker} Price: Historical Data ({ticker}) and Random Walk Simulation Aggregates')
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
