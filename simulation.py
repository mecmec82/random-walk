import streamlit as st
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import time
import math

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Market Random Walk Simulation (yfinance)", layout="wide")

st.title("Market Price Random Walk Simulation")
st.write("Simulate multiple future price movements using random walks based on historical volatility, using free data via yfinance.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

ticker = st.sidebar.text_input("Ticker Symbol (e.g., SPY, BTC-USD)", 'BTC-USD').upper()

# Number input for historical days used for ANALYSIS
historical_days_requested = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=100, value=900, step=100)

simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=200, step=10)

st.sidebar.write("Data fetched using yfinance (Yahoo Finance).")
st.sidebar.write("Note: Yahoo Finance data may have occasional inaccuracies or downtime.")

# --- Sidebar Results Display Area ---
st.sidebar.header("Simulation Insights")
sidebar_placeholder = st.sidebar.empty() # Create a placeholder to update results


# --- Helper function to fetch historical data using yfinance ---
@st.cache_data # Cache data fetching results to avoid re-fetching on rerun
def fetch_historical_data(ticker, num_days):
    """
    Fetches historical data for the last num_days using yfinance.
    Returns a pandas Series of closing prices with datetime index.
    Includes basic filtering for positive prices.
    """
    st.info(f"Fetching historical data for {ticker}...")

    # Determine start date: go back enough calendar days to cover num_days *trading* days
    # Using 1.5x as a conservative estimate for stocks (approx 252 trading days/year)
    # For crypto (7 days/week), 1x is enough, but 1.5x is safe for both.
    # We need to fetch *at least* num_days, potentially more if available,
    # so we can select the last N for analysis *and* later select a tail for display.
    # Fetching more than requested is generally fine for caching.
    # Let's fetch enough to satisfy the *maximum* reasonable request (e.g., 5 years ~ 1300 days)
    # or just rely on yfinance's default fetch behavior (which is usually a lot)
    # and then take the tail for the requested `num_days` *for analysis*.
    # The buffer in days should be based on the *requested* days, not a fixed large number,
    # to keep caching efficient if requested days is changed.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days * 2) # Increased buffer slightly


    try:
        # yf.download returns a DataFrame
        # It fetches data between start and end dates.
        # We will then take the last `num_days` of *that fetched data* for analysis.
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series() # Return empty Series

        # We need the 'Close' price and ensure it's sorted by date (ascending)
        # yfinance usually returns data sorted ascending, but explicit sort is safe
        historical_data_close = data['Close'].sort_index()

        # Filter out any non-positive prices proactively
        original_len = len(historical_data_close)
        historical_data_close = historical_data_close[historical_data_close > 0]
        if len(historical_data_close) < original_len:
             st.warning(f"Filtered out {original_len - len(historical_data_close)} rows with non-positive prices.")


        # Return the full fetched & filtered Series.
        # We will take the tail for analysis *after* fetching.
        # This allows the slider to control the display tail without re-fetching the whole period.
        st.success(f"Successfully fetched {len(historical_data_close)} trading days.")
        return historical_data_close # Return the full Series here


    except Exception as e:
        st.error(f"Error fetching data for {ticker} using yfinance: {e}")
        st.error("Please check the ticker symbol and your internet connection.")
        return pd.Series()


# --- Main Simulation Logic ---
# This block runs whenever any Streamlit input changes, including the slider.
# But the @st.cache_data decorator prevents re-fetching if fetch_historical_data's args haven't changed.

# --- Fetch Historical Data (Cached) ---
# Fetch the potentially larger dataset needed for analysis,
# based on historical_days_requested
full_historical_data = fetch_historical_data(
    ticker=ticker,
    num_days=historical_days_requested # Fetch data for at least this many days
)

# Select the specific subset for ANALYSIS (the most recent historical_days_requested days)
# This is the data used for calculating mean/volatility
historical_data_close_analyzed = full_historical_data.tail(historical_days_requested)


# --- Check if we have enough data for analysis ---
if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
     # Display initial state or error if not enough data even before button click
     st.warning("Enter parameters and click 'Run Simulation'. Not enough historical data available for analysis.")
     st.stop() # Stop execution if we can't even start


# --- Calculate Historical Returns and Volatility ---
# This calculation happens whenever inputs change, which includes the slider.
# We could cache this too if it were very expensive, but it's usually fast.
with st.spinner("Calculating historical statistics..."):
    log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

    if len(log_returns) < 1:
        st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after dropping NaNs. Need at least 2 consecutive valid prices.")
        st.stop()

    mean_daily_log_return = log_returns.mean()
    daily_log_volatility = log_returns.std()

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

# --- Historical Analysis Results (Display only if button is clicked) ---
# Move this section inside the button click to only show after simulation runs
# st.subheader("Historical Analysis Results")
# st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}**:")
# st.info(f"Calculated Mean Daily Log Return: `{mean_float:.6f}`")
# st.info(f"Calculated Daily Log Volatility: `{volatility_float:.6f}`")


# --- Run Simulation (Only when button is clicked) ---
# The simulation itself is still triggered by the button, as it's a heavy process
if st.button("Run Simulation"):

    # --- Calculate Simulation Aggregates ---
    # The simulation and aggregate calculation run *once* per button click
    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):

        # Get start price from the data used for analysis
        try:
            raw_start_price_value = historical_data_close_analyzed.iloc[-1]
            start_price = float(raw_start_price_value)
        except (TypeError, ValueError) as e:
             st.error(f"Unexpected value or type for last historical price before conversion: {e}")
             st.info(f"Last price value: {raw_start_price_value}, type: {type(raw_start_price_value)}")
             start_price = np.nan

        if not np.isfinite(start_price):
             st.error(f"Last historical price ({start_price}) is not a finite number after conversion. Cannot start simulation.")
             st.stop()


        # Prepare dates for simulation results plotting (based on analysis dates)
        historical_dates_analysis = historical_data_close_analyzed.index
        last_historical_date_analysis = historical_dates_analysis.max()

        simulated_dates = pd.DatetimeIndex([])
        sim_path_length = 0

        try:
            simulated_dates = pd.date_range(start=last_historical_date_analysis, periods=simulation_days + 1, freq='B')[1:]
            if len(simulated_dates) != simulation_days:
                 st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date_analysis.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length will be adjusted.")
            sim_path_length = len(simulated_dates) + 1
        except Exception as date_range_error:
             st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
             st.stop()

        plot_sim_dates = pd.DatetimeIndex([])
        if len(simulated_dates) > 0 and sim_path_length > 0:
            last_historical_date_index = pd.DatetimeIndex([last_historical_date_analysis])
            plot_sim_dates = last_historical_date_index.append(simulated_dates)
            plot_sim_dates = pd.DatetimeIndex(plot_sim_dates)
            if len(plot_sim_dates) != sim_path_length:
                 st.error(f"Mismatch between calculated path length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates)}). Cannot plot.")
                 st.stop()
        else:
             st.error("Skipping simulation as future dates could not be generated.")
             st.stop()


        # --- Run Multiple Simulations (Inner loop) ---
        all_simulated_paths = []
        for _ in range(num_simulations):
            simulated_log_returns = np.random.normal(
                loc=mean_float,
                scale=volatility_float,
                size=simulation_days
            )
            simulated_price_path = np.zeros(sim_path_length)
            simulated_price_path[0] = start_price
            for j in range(1, sim_path_length):
                 if j - 1 < len(simulated_log_returns):
                      if np.isfinite(simulated_price_path[j-1]):
                           simulated_price_path[j] = simulated_price_path[j-1] * np.exp(simulated_log_returns[j-1])
                      else:
                           simulated_price_path[j] = np.nan
                 else:
                      simulated_price_path[j] = simulated_price_path[j-1]

            all_simulated_paths.append(simulated_price_path)


        # --- Calculate Median, Mean, Standard Deviation Across Simulations ---
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
                     # No band arrays created if no valid points
                else:
                    upper_band = mean_prices + std_dev_prices
                    lower_band = mean_prices - std_dev_prices
                    upper_band[~valid_agg_points] = np.nan
                    lower_band[~valid_agg_points] = np.nan

                    final_prices = [path[-1] for path in all_simulated_paths if np.isfinite(path[-1])]

            except Exception as e:
                st.error(f"Error calculating aggregate statistics: {e}")
                # Ensure aggregate arrays are empty if calculation fails
                median_prices = np.array([])
                mean_prices = np.array([])
                std_dev_prices = np.array([])
                upper_band = np.array([])
                lower_band = np.array([])
                final_prices = []
        else:
            st.warning("Simulation paths have inconsistent or zero lengths. Cannot calculate aggregate statistics.")


        # --- Calculate Sidebar Results Here (After Aggregates) ---
        # These calculations happen once per button click
        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        # Ensure last historical price is available and finite (already checked start_price)
        if len(historical_data_close_analyzed) > 0:
             last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1])
             if np.isfinite(last_historical_price_scalar) and last_historical_price_scalar > 0:
                  final_upper_price = upper_band[-1] if len(upper_band) > 0 and np.isfinite(upper_band[-1]) else np.nan # Ensure final aggregate is finite
                  final_lower_price = lower_band[-1] if len(lower_band) > 0 and np.isfinite(lower_band[-1]) else np.nan # Ensure final aggregate is finite

                  # Percentage Delta to +1 Std Dev
                  if np.isfinite(final_upper_price):
                       delta_upper_pct = ((final_upper_price - last_historical_price_scalar) / last_historical_price_scalar) * 100

                  # Percentage Delta to -1 Std Dev
                  if np.isfinite(final_lower_price):
                       delta_lower_pct = ((final_lower_price - last_historical_price_scalar) / last_historical_price_scalar) * 100

                  # Risk/Reward Ratio
                  if np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
                       potential_reward = delta_upper_pct
                       potential_risk_abs = -delta_lower_pct # Absolute magnitude of downside movement

                       if potential_risk_abs > 0:
                            risk_reward_ratio = potential_reward / potential_risk_abs
                       elif potential_risk_abs == 0:
                            risk_reward_ratio = np.inf if potential_reward > 0 else np.nan


    # --- Display Sidebar Results (Only when button clicked) ---
    with sidebar_placeholder.container():
         st.subheader("Key Forecasts")
         if np.isfinite(delta_upper_pct):
              st.write(f"Expected movement to +1 Std Dev End: **{delta_upper_pct:.2f}%**")
         else:
              st.write("Expected movement to +1 Std Dev End: **N/A**")

         if np.isfinite(delta_lower_pct):
              st.write(f"Expected movement to -1 Std Dev End: **{delta_lower_pct:.2f}%**")
         else:
              st.write("Expected movement to -1 Std Dev End: **N/A**")

         st.subheader("Risk/Reward")
         if np.isfinite(risk_reward_ratio):
              if risk_reward_ratio == np.inf:
                   st.write("Ratio (+1 Gain : -1 Loss): **Infinite**")
              else:
                   st.write(f"Ratio (+1 Gain : -1 Loss): **{risk_reward_ratio:.2f} : 1**")
         elif np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
               st.write("Ratio (+1 Gain : -1 Loss): **Undetermined / Favorable Downside**")
         else:
              st.write("Ratio (+1 Gain : -1 Loss): **N/A**")

         # Also add basic info about number of simulation runs to sidebar
         st.write(f"*(Based on {num_simulations} runs)*")

    # --- END Display Sidebar Results ---


    # --- Display Historical Analysis Results (Only when button clicked) ---
    st.subheader("Historical Analysis Results")
    st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}**:")
    # Use the already calculated floats for display
    st.info(f"Calculated Mean Daily Log Return: `{mean_float:.6f}`")
    st.info(f"Calculated Daily Log Volatility: `{volatility_float:.6f}`")


    # --- Plotting ---
    st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")

    # Define the number of historical days to display on the plot
    # Use the slider value here
    historical_days_to_display = st.sidebar.slider(
        "Historical Days to Display on Plot",
        min_value=100, # Minimum display days
        max_value=len(historical_data_close_analyzed), # Max is the length of data fetched for analysis
        value=min(historical_days_requested, len(historical_data_close_analyzed)), # Default to requested days, capped by available data
        step=10
    )

    # Select the tail of the historical data *specifically for plotting*
    historical_data_to_plot = historical_data_close_analyzed.tail(historical_days_to_display)
    historical_dates_to_plot = historical_data_to_plot.index


    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot Historical Data (using the filtered data for display)
    if not historical_data_to_plot.empty:
        ax.plot(historical_dates_to_plot, historical_data_to_plot.values, label=f'Historical {ticker} Price ({len(historical_data_to_plot)} days displayed)', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    # Ensure we have dates for plotting and that median_prices is not empty/all NaN
    # These are based on the *full* analysis period's end date and the simulation days
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

            # --- Add Labels at the end ---
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


    # Set plot title to reflect displayed historical range vs analysis range
    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)} days displayed, {len(historical_data_close_analyzed)} days analyzed) and Random Walk Simulation Aggregates'
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)

    try:
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generating plot: {e}")

    plt.close(fig)

    # --- Display Final Results (Main Area) ---
    st.subheader("Simulation Results Overview")
    if len(historical_data_close_analyzed) > 0:
        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1])
        st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${last_historical_price_scalar:.2f}**")

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
