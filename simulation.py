import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import time
import math

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Market Random Walk Simulation (yfinance Debug)", layout="wide")

st.title("Market Price Random Walk Simulation")
st.write("Simulate multiple future price movements using random walks based on historical volatility, using free data via yfinance.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

ticker = st.sidebar.text_input("Ticker Symbol (e.g., SPY, BTC-USD)", 'BTC-USD').upper()

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
    Includes basic filtering for positive prices.
    """
    st.info(f"Fetching historical data for {ticker}...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days * 1.5) # Buffer for weekends/holidays

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series()

        historical_data_close = data['Close'].sort_index()

        # Filter out any non-positive prices proactively
        original_len = len(historical_data_close)
        historical_data_close = historical_data_close[historical_data_close > 0]
        if len(historical_data_close) < original_len:
             st.warning(f"Filtered out {original_len - len(historical_data_close)} rows with non-positive prices.")

        # Take the last 'num_days' trading days from the filtered data
        historical_data_close_analyzed = historical_data_close.tail(num_days)

        if len(historical_data_close_analyzed) < num_days:
             st.warning(f"Only {len(historical_data_close_analyzed)} trading days available for analysis after fetching, filtering, and selecting the last {num_days} days.")
        else:
             st.success(f"Successfully fetched {len(historical_data_close_analyzed)} trading days for analysis.")

        # --- DEBUGGING: Print last 5 rows of the data used for analysis ---
        st.subheader("Historical Data Used for Analysis (Last 5 rows):")
        if not historical_data_close_analyzed.empty:
             st.dataframe(historical_data_close_analyzed.tail(5))
        else:
             st.warning("Historical data is empty after filtering.")
        # --- END DEBUGGING ---


        return historical_data_close_analyzed

    except Exception as e:
        st.error(f"Error fetching data for {ticker} using yfinance: {e}")
        st.error("Please check the ticker symbol and your internet connection.")
        return pd.Series()


# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):

    # --- Fetch Historical Data ---
    historical_data_close_analyzed = fetch_historical_data(
        ticker=ticker,
        num_days=historical_days_requested
    )

    # Ensure we have enough data AFTER the fetch attempts and filtering
    if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
         st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available for analysis after fetching and filtering. Need at least 2 days with valid prices.")
         st.stop()


    # --- Calculate Historical Returns and Volatility ---
    with st.spinner("Calculating historical statistics..."):
        log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

        if len(log_returns) < 1:
            st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after dropping NaNs. Need at least 2 consecutive valid prices.")
            st.stop()

        mean_daily_log_return = log_returns.mean()
        daily_log_volatility = log_returns.std()

        # --- Explicitly cast to float and check for finite numbers before formatting ---
        try:
            mean_float = float(mean_daily_log_return)
            volatility_float = float(daily_log_volatility)
        except (TypeError, ValueError) as e:
             st.error(f"Unexpected value or type for calculated mean or volatility: {e}")
             st.info(f"Mean value: {mean_daily_log_return}, Mean type: {type(mean_daily_log_return)}")
             st.info(f"Volatility value: {daily_log_volatility}, Volatility type: {type(daily_log_volatility)}")
             st.stop()


        if not np.isfinite(mean_float) or not np.isfinite(volatility_float):
             st.error(f"Could not calculate finite mean or volatility from historical data.")
             st.info(f"Calculated mean: {mean_float}, calculated volatility: {volatility_float}")
             st.stop()


        st.subheader("Historical Analysis Results")
        st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}**:")
        st.info(f"Calculated Mean Daily Log Return: `{mean_float:.6f}`")
        st.info(f"Calculated Daily Log Volatility: `{volatility_float:.6f}`")


    # --- Prepare Dates for Plotting (Historical + Simulated) ---
    historical_dates = historical_data_close_analyzed.index
    last_historical_date = historical_dates.max()

    # Generate future dates for the simulation
    simulated_dates = pd.DatetimeIndex([])
    sim_path_length = 0

    try:
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:]

        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length will be adjusted.")

        sim_path_length = len(simulated_dates) + 1


    except Exception as date_range_error:
         st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
         simulated_dates = pd.DatetimeIndex([])
         sim_path_length = 0


    # Combine historical last date with simulated dates for plotting the simulation paths
    plot_sim_dates = pd.DatetimeIndex([])
    if len(simulated_dates) > 0 and sim_path_length > 0:
        last_historical_date_index = pd.DatetimeIndex([last_historical_date])
        plot_sim_dates = last_historical_date_index.append(simulated_dates)
        plot_sim_dates = pd.DatetimeIndex(plot_sim_dates)
        if len(plot_sim_dates) != sim_path_length:
             st.error(f"Mismatch between calculated path length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates)}). Cannot plot.")
             plot_sim_dates = pd.DatetimeIndex([])

    else:
        st.warning("Skipping simulation date axis generation.")


    # --- Run Multiple Simulations ---
    all_simulated_paths = []
    # Only proceed if we have enough historical data and successfully generated simulation dates
    if sim_path_length > 0 and len(historical_data_close_analyzed) > 0:
        with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):
            # Get start price and explicitly cast to float for safety
            try:
                raw_start_price_value = historical_data_close_analyzed.iloc[-1]
                start_price = float(raw_start_price_value) # Explicitly cast to float here
            except (TypeError, ValueError) as e:
                 st.error(f"Unexpected value or type for last historical price before conversion: {e}")
                 st.info(f"Last price value: {raw_start_price_value}, type: {type(raw_start_price_value)}")
                 start_price = np.nan # Set to NaN if conversion fails


            # Check if start_price is a finite number AFTER the float conversion
            # This line corresponds to line 178 in your traceback
            if not np.isfinite(start_price): # This check should now receive a float
                 st.error(f"Last historical price ({start_price}) is not a finite number after conversion. Cannot start simulation.")
                 all_simulated_paths = [] # Clear paths so simulation is skipped below
            else:
                for _ in range(num_simulations):
                    simulated_log_returns = np.random.normal(
                        loc=mean_float,
                        scale=volatility_float,
                        size=simulation_days
                    )

                    simulated_price_path = np.zeros(sim_path_length)
                    if sim_path_length > 0:
                        simulated_price_path[0] = start_price # Use the confirmed float start_price

                        for j in range(1, sim_path_length):
                             if j - 1 < len(simulated_log_returns):
                                  if np.isfinite(simulated_price_path[j-1]):
                                       simulated_price_path[j] = simulated_price_path[j-1] * np.exp(simulated_log_returns[j-1])
                                  else:
                                       simulated_price_path[j] = np.nan
                             else:
                                  simulated_price_path[j] = simulated_price_path[j-1]


                    all_simulated_paths.append(simulated_price_path)
    else:
        st.warning("Skipping simulations as future dates could not be generated or historical data is missing.")


    # --- Calculate Median, Mean, Standard Deviation, and Final Prices ---
    median_prices = np.array([])
    mean_prices = np.array([])
    std_dev_prices = np.array([])
    upper_band = np.array([])
    lower_band = np.array([])
    final_prices = []

    if len(all_simulated_paths) > 0 and sim_path_length > 0 and all(len(path) == sim_path_length for path in all_simulated_paths):
        try:
            all_simulated_paths_np = np.vstack(all_simulated_paths)
            prices_at_each_step = all_simulated_paths_np.T

            median_prices = np.nanmedian(prices_at_each_step, axis=1)
            mean_prices = np.nanmean(prices_at_each_step, axis=1)
            std_dev_prices = np.nanstd(prices_at_each_step, axis=1)

            valid_agg_points = np.isfinite(mean_prices) & np.isfinite(std_dev_prices) & np.isfinite(median_prices)
            if not valid_agg_points.any():
                 st.warning("Calculated simulation aggregates contain non-finite values (NaN or Inf) for all steps. This can happen if many simulation paths diverged early.")
                 median_prices[:] = np.nan
                 mean_prices[:] = np.nan
                 std_dev_prices[:] = np.nan
                 upper_band = np.array([])
                 lower_band = np.array([])
            else:
                upper_band = mean_prices + std_dev_prices
                lower_band = mean_prices - std_dev_prices
                upper_band[~valid_agg_points] = np.nan
                lower_band[~valid_agg_points] = np.nan

                final_prices = [path[-1] for path in all_simulated_paths if np.isfinite(path[-1])]


        except Exception as e:
            st.error(f"Error calculating aggregate statistics: {e}")


    elif len(all_simulated_paths) > 0:
        st.warning("Simulation paths have inconsistent or zero lengths. Cannot calculate aggregate statistics.")
    else:
        st.warning("No simulation paths were generated successfully.")


    # --- Plotting ---
    st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")

    fig, ax = plt.subplots(figsize=(14, 7))

    if not historical_data_close_analyzed.empty:
        ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    if len(plot_sim_dates) > 0 and len(median_prices) == len(plot_sim_dates) and np.isfinite(median_prices).any():

         valid_plot_indices = np.isfinite(median_prices)
         plot_sim_dates_valid = plot_sim_dates[valid_plot_indices]
         median_prices_valid = median_prices[valid_plot_indices]
         upper_band_valid = upper_band[valid_plot_indices] if len(upper_band) == len(median_prices) else np.array([])
         lower_band_valid = lower_band[valid_plot_indices] if len(lower_band) == len(median_prices) else np.array([])


         if len(plot_sim_dates_valid) > 0:
            ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations} runs)', color='red', linestyle='-', linewidth=2)

            if len(upper_band_valid) == len(plot_sim_dates_valid) and len(lower_band_valid) == len(plot_sim_dates_valid):
                 ax.fill_between(plot_sim_dates_valid, lower_band_valid, upper_band_valid, color='orange', alpha=0.3, label='+/- 1 Std Dev Band')
            else:
                 st.warning("Length mismatch for plotting valid std dev band segments. Band will not be plotted.")

            if np.isfinite(median_prices[-1]):
                 final_date = plot_sim_dates[-1]

                 ax.text(final_date, median_prices[-1], f" ${median_prices[-1]:.2f}",
                         color='red', fontsize=10, ha='left', va='center', weight='bold')

                 if len(upper_band) > 0 and np.isfinite(upper_band[-1]):
                      ax.text(final_date, upper_band[-1], f" ${upper_band[-1]:.2f}",
                              color='darkorange', fontsize=9, ha='left', va='bottom')

                 if len(lower_band) > 0 and np.isfinite(lower_band[-1]):
                       ax.text(final_date, lower_band[-1], f" ${lower_band[-1]:.2f}",
                              color='darkorange', fontsize=9, ha='left', va='top')

            else:
                st.warning("Last simulated data point in median is not finite, skipping end labels.")

            ax.legend()
         else:
              st.warning("No finite aggregate simulation data points available to plot the median line/band.")


    elif len(all_simulated_paths) > 0:
        st.warning("Could not plot simulation aggregates. Data length issues or all aggregate values are non-finite.")
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
        # --- CRITICAL FIX HERE ---
        # Explicitly cast to float() the value from .iloc[-1] before formatting
        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1])
        st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${last_historical_price_scalar:.2f}**")
        # --- END CRITICAL FIX ---

    if len(median_prices) > 0 and np.isfinite(median_prices[-1]):
         st.write(f"Ran **{num_simulations}** simulations.")
         if len(simulated_dates) > 0:
              st.write(f"Simulated Ending Prices (after {len(simulated_dates)} steps, {simulated_dates[-1].strftime('%Y-%m-%d')}):")
              st.write(f"- Median: **${median_prices[-1]:.2f}**")
              if len(mean_prices) > 0 and np.isfinite(mean_prices[-1]):
                   st.write(f"- Mean: ${mean_prices[-1]:.2f}")
              if len(std_dev_prices) > 0 and np.isfinite(std_dev_prices[-1]):
                   st.write(f"- Std Dev: ${std_dev_prices[-1]:.2f}")
              if len(lower_band) > 0 and len(upper_band) > 0 and np.isfinite(lower_band[-1]) and np.isfinite(upper_band[-1]):
                   st.write(f"- +/- 1 Std Dev Range: [${lower_band[-1]:.2f}, ${upper_band[-1]:.2f}]")
              else:
                   st.warning("Ending Std Dev band values are non-finite.")


              if final_prices:
                  st.write(f"- Actual Min Ending Price: ${np.min(final_prices):.2f}")
                  st.write(f"- Actual Max Ending Price: ${np.max(final_prices):.2f}")
              else:
                   st.warning("No finite final prices from simulated paths to calculate min/max.")

         else:
              st.warning("Simulated dates were not generated successfully.")

    elif len(all_simulated_paths) > 0:
        st.warning("Simulation paths were generated, but ending aggregate statistics are non-finite.")
    else:
         st.warning("No simulation results to display.")
