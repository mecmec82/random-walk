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

# --- Initialize Session State for Simulation Results ---
# This ensures simulation results persist across reruns caused by slider changes
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

ticker = st.sidebar.text_input("Ticker Symbol (e.g., SPY, BTC-USD)", 'BTC-USD').upper()

# Number input for historical days used for ANALYSIS
# This input triggers a rerun and cached data fetch/analysis
historical_days_requested = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=100, value=900, step=100)

simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=200, step=10)

st.sidebar.write("Data fetched using yfinance (Yahoo Finance).")
st.sidebar.write("Note: Yahoo Finance data may have occasional inaccuracies or downtime.")

# --- Sidebar Results Display Area ---
st.sidebar.header("Simulation Insights")
sidebar_placeholder = st.sidebar.empty() # Create a placeholder to update results


# --- Helper function to fetch historical data using yfinance ---
# Cached function: runs only when ticker or historical_days_requested changes
@st.cache_data(ttl=3600) # Cache data fetching results for 1 hour (3600 seconds)
def fetch_historical_data(ticker, num_days):
    """
    Fetches historical data for a period covering at least num_days
    using yfinance. Returns a pandas Series of closing prices with datetime index.
    Includes basic filtering for positive prices.
    """
    st.info(f"Fetching historical data for {ticker} to cover at least {num_days} days...")

    # Determine start date: go back enough calendar days to cover num_days *trading* days
    # Using 2x as a conservative estimate for stocks (approx 252 trading days/year)
    # to ensure we fetch enough history even for longer analysis periods.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_days * 2) # Increased buffer


    try:
        # yf.download fetches data between start and end dates.
        # We will then take the last `num_days_requested` of *that fetched data* for analysis,
        # and a potentially smaller tail for display.
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series() # Return empty Series

        # We need the 'Close' price and ensure it's sorted by date (ascending)
        historical_data_close = data['Close'].sort_index()

        # Filter out any non-positive prices proactively
        original_len = len(historical_data_close)
        historical_data_close = historical_data_close[historical_data_close > 0]
        if len(historical_data_close) < original_len:
             st.warning(f"Filtered out {original_len - len(historical_data_close)} rows with non-positive prices.")


        # Return the full fetched & filtered Series.
        # We will take the tail for analysis *and* display *after* fetching.
        # This allows the slider to control the display tail without re-fetching the whole period.
        st.success(f"Successfully fetched {len(historical_data_close)} trading days.")
        return historical_data_close # Return the full Series here


    except Exception as e:
        st.error(f"Error fetching data for {ticker} using yfinance: {e}")
        st.error("Please check the ticker symbol and your internet connection.")
        return pd.Series()


# --- Fetch Historical Data (Cached - runs on initial load and input changes) ---
# This block runs on every rerun, but the fetch_historical_data function itself is cached.
# It fetches enough data to satisfy the historical_days_requested plus a buffer.
full_historical_data = fetch_historical_data(
    ticker=ticker,
    num_days=historical_days_requested # Fetch data for at least this many days + buffer
)

# Select the specific subset for ANALYSIS (the most recent historical_days_requested days)
# This is the data used downstream for calculating mean/volatility and starting the simulation.
# This calculation runs on every rerun, but uses the cached full_historical_data.
historical_data_close_analyzed = full_historical_data.tail(historical_days_requested)


# --- Check if we have enough data for analysis ---
# This check runs on initial load and input changes (before button click)
if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
     # Display initial state or error if not enough data
     st.warning("Enter parameters and click 'Run Simulation'. Not enough historical data available for analysis.")
     # Stop here if data is insufficient, so we don't try to create a slider or calculate stats with invalid range
     st.stop()


# --- Calculate Historical Returns and Volatility ---
# This calculation happens whenever inputs change, which includes the slider.
# It uses `historical_data_close_analyzed` which is NOT affected by the slider.
# It depends on the cached fetch result, so it's efficient unless inputs change.
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

# --- Historical Analysis Results (Display) ---
# This section displays the analysis results based on historical_days_requested
# It runs on every rerun.
st.subheader("Historical Analysis Results")
st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}**:")
st.info(f"Calculated Mean Daily Log Return: `{mean_float:.6f}`")
st.info(f"Calculated Daily Log Volatility: `{volatility_float:.6f}`")


# --- Add Slider for Displayed Historical Days ---
# This slider is defined *outside* the button block so it appears immediately.
# Its max_value is based on the actual number of days successfully fetched and available for analysis.
max_display_days = len(historical_data_close_analyzed)
# Set default value: requested days, capped by available, min 100 unless less data is available
default_display_days = min(historical_days_requested, max_display_days)
default_display_days = max(100, default_display_days) if max_display_days >= 100 else max_display_days


historical_days_to_display = st.sidebar.slider(
    "Historical Days to Display on Plot",
    min_value=min(100, max_display_days), # Ensure min_value doesn't exceed available data
    max_value=max_display_days,
    value=default_display_days,
    step=10,
    help="Slide to change the number of historical trading days shown on the chart. Does not affect analysis period."
)

# --- Define the plotting function ---
# This function will be called *outside* the button block to redraw the plot
# It takes the full historical data, the slider value, and the simulation results from session state
def plot_simulation(full_historical_data, historical_days_to_display, simulation_results):
    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Select data for plotting using the slider value ---
    # Use the full fetched data and take the tail based on the slider value
    historical_data_to_plot = full_historical_data.tail(historical_days_to_display)
    historical_dates_to_plot = historical_data_to_plot.index


    # Plot Historical Data (using the filtered data for display)
    if not historical_data_to_plot.empty:
        ax.plot(historical_dates_to_plot, historical_data_to_plot.values, label=f'Historical {ticker} Price ({len(historical_data_to_plot)} days displayed)', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    # Only plot simulation results if they exist in session state
    if simulation_results is not None:
         # Unpack results from the session state dictionary
         plot_sim_dates = simulation_results['plot_sim_dates']
         median_prices = simulation_results['median_prices']
         upper_band = simulation_results['upper_band']
         lower_band = simulation_results['lower_band']
         num_simulations_ran = simulation_results['num_simulations_ran'] # Get the actual number of sims ran

         # Check if plotting aggregates is possible based on the simulation results
         if len(plot_sim_dates) > 0 and len(median_prices) == len(plot_sim_dates) and np.isfinite(median_prices).any():

              valid_plot_indices = np.isfinite(median_prices)
              plot_sim_dates_valid = plot_sim_dates[valid_plot_indices]
              median_prices_valid = median_prices[valid_plot_indices]
              # Filter bands based on the same indices (check length consistency)
              upper_band_valid = upper_band[valid_plot_indices] if len(upper_band) == len(median_prices) else np.array([])
              lower_band_valid = lower_band[valid_plot_indices] if len(lower_band) == len(median_prices) else np.array([])


              if len(plot_sim_dates_valid) > 0:
                 ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations_ran} runs)', color='red', linestyle='-', linewidth=2)

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


         else:
             st.warning("Could not plot simulation aggregates. Data length issues or all aggregate values are non-finite.")
             # Still plot historical data even if simulation plot fails
             # The historical plot is already done above
    # else: # simulation_results is None, so no simulation has been run yet or it failed
        # st.info("Click 'Run Simulation' to see forecasts.") # Optional message


    # Set plot title to reflect displayed historical range vs analysis range
    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)} days displayed, {len(historical_data_close_analyzed)} days analyzed) and Random Walk Simulation Aggregates'
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)

    # Return the figure object
    return fig


# --- Define functions to display results ---
# These functions read from session state and display results

def display_sidebar_results(simulation_results, placeholder):
     # Clear previous sidebar results
     placeholder.empty()
     with placeholder.container():
         st.subheader("Key Forecasts")
         if simulation_results is not None:
              delta_upper_pct = simulation_results['delta_upper_pct']
              delta_lower_pct = simulation_results['delta_lower_pct']
              risk_reward_ratio = simulation_results['risk_reward_ratio']
              num_simulations_ran = simulation_results['num_simulations_ran']

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
              # Handle specific NaN cases where delta is finite but ratio is not
              elif np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
                   st.write("Ratio (+1 Gain : -1 Loss): **Undetermined / Favorable Downside**")
              else:
                   st.write("Ratio (+1 Gain : -1 Loss): **N/A**")

              st.write(f"*(Based on {num_simulations_ran} runs)*")
         else:
              # Display initial message if no simulation run yet
              st.info("Click 'Run Simulation' to see forecasts.")


def display_main_results_overview(simulation_results, historical_data_close_analyzed):
     st.subheader("Simulation Results Overview")
     if len(historical_data_close_analyzed) > 0:
         # Ensure last historical price is available and finite for display
         try:
             last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1])
             if np.isfinite(last_historical_price_scalar):
                  st.write(f"**Last Historical Price** ({historical_data_close_analyzed.index[-1].strftime('%Y-%m-%d')}): **${last_historical_price_scalar:.2f}**")
             else:
                  st.warning("Last historical price is not finite, cannot display.")
         except Exception:
              st.warning("Could not retrieve or display last historical price.")


     # Display other results here, excluding the ones moved to sidebar
     if simulation_results is not None:
          median_prices = simulation_results['median_prices']
          mean_prices = simulation_results['mean_prices']
          std_dev_prices = simulation_results['std_dev_prices']
          lower_band = simulation_results['lower_band']
          upper_band = simulation_results['upper_band']
          final_prices = simulation_results['final_prices'] # List of actual final prices
          simulated_dates = simulation_results['simulated_dates'] # Dates for the simulation period


          # Display this section only if simulation aggregates were successfully calculated and are finite at the end
          if len(median_prices) > 0 and np.isfinite(median_prices[-1]):
               st.write(f"Ran **{simulation_results['num_simulations_ran']}** simulations.")
               if len(simulated_dates) > 0: # Final check for data availability
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

                    if final_prices: # Check if the list of finite final prices was populated
                        st.write(f"- Actual Min Ending Price: ${np.min(final_prices):.2f}")
                        st.write(f"- Actual Max Ending Price: ${np.max(final_prices):.2f}")
                    else:
                         st.warning("No finite final prices from simulated paths to calculate min/max.")

               else:
                    st.warning("Simulated dates were not generated successfully.")

          elif 'all_simulated_paths' in simulation_results and len(simulation_results['all_simulated_paths']) > 0:
              st.warning("Simulation paths were generated, but ending aggregate statistics are non-finite.")
          else:
               st.warning("No simulation results to display.")


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis, Slider Definition) runs on every rerun.

# --- Button to Run Simulation ---
# This block runs ONLY when the button is clicked
if st.button("Run Simulation"):
    # Clear previous sidebar results visual (placeholder)
    sidebar_placeholder.empty()

    # --- Calculate Simulation Aggregates (Heavy Computation) ---
    # This happens once per button click
    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days..."):

        # Get start price from the data used for analysis
        try:
            raw_start_price_value = historical_data_close_analyzed.iloc[-1]
            start_price = float(raw_start_price_value)
        except (TypeError, ValueError) as e:
             st.error(f"Unexpected value or type for last historical price before conversion: {e}")
             start_price = np.nan # Set to NaN if conversion fails

        if not np.isfinite(start_price):
             st.error(f"Last historical price ({start_price}) is not a finite number after conversion. Cannot start simulation.")
             # Set session state to None to indicate simulation failed/skipped
             st.session_state.simulation_results = None
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
             st.session_state.simulation_results = None
             st.stop()

        plot_sim_dates = pd.DatetimeIndex([])
        if len(simulated_dates) > 0 and sim_path_length > 0:
            last_historical_date_index = pd.DatetimeIndex([last_historical_date_analysis])
            plot_sim_dates = last_historical_date_index.append(simulated_dates)
            plot_sim_dates = pd.DatetimeIndex(plot_sim_dates)
            if len(plot_sim_dates) != sim_path_length:
                 st.error(f"Mismatch between calculated path length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates)}). Cannot plot.")
                 st.session_state.simulation_results = None
                 st.stop()
        else:
             st.error("Skipping simulation as future dates could not be generated.")
             st.session_state.simulation_results = None
             st.stop()


        # --- Run Multiple Simulations (Inner loop) ---
        all_simulated_paths = [] # Store individual paths temporarily if needed for debugging, but not strictly necessary for aggregates
        # The aggregate calculation uses the definition of num_simulations directly

        # Use pre-calculated finite mean/volatility
        loc_sim = mean_float
        scale_sim = volatility_float

        # Generate all random returns for all simulations at once for efficiency
        all_simulated_log_returns = np.random.normal(
             loc=loc_sim,
             scale=scale_sim,
             size=(num_simulations, simulation_days) # Shape is (num_simulations, num_steps)
        )

        # Calculate simulated price paths efficiently using cumulative product
        # Start with a 2D array where each row is the start price, shape (num_simulations, 1)
        start_prices_array = np.full((num_simulations, 1), start_price)

        # Calculate the price changes from returns: exp(log_return)
        price_changes = np.exp(all_simulated_log_returns) # Shape (num_simulations, simulation_days)

        # Calculate cumulative product of price changes + initial price
        # Need to prepend 1.0 to the price_changes for the cumulative product
        # axis=1 means cumulative product along each row (each simulation path)
        cumulative_price_changes = np.cumprod(np.concatenate((np.ones((num_simulations, 1)), price_changes), axis=1), axis=1) # Shape (num_simulations, simulation_days + 1)

        # Multiply by the start price
        all_simulated_paths_np = start_prices_array * cumulative_price_changes # Shape (num_simulations, simulation_days + 1)

        # Ensure the shape matches sim_path_length (handle potential date trimming)
        # If sim_path_length < simulation_days + 1, we need to truncate the simulated paths
        if all_simulated_paths_np.shape[1] > sim_path_length:
             st.warning(f"Simulation paths calculated with length {all_simulated_paths_np.shape[1]}, but expected length is {sim_path_length} due to date generation. Truncating paths.")
             all_simulated_paths_np = all_simulated_paths_np[:, :sim_path_length]
             # Recalculate final prices list from the truncated paths
             final_prices_list_raw = [path[-1] for path in all_simulated_paths_np]
        else:
             # Calculate final prices list from the full paths
             final_prices_list_raw = [path[-1] for path in all_simulated_paths_np]


        # --- Calculate Median, Mean, Standard Deviation Across Simulations ---
        # Transpose to have prices at each step in columns
        prices_at_each_step = all_simulated_paths_np.T

        median_prices = np.nanmedian(prices_at_each_step, axis=1)
        mean_prices = np.nanmean(prices_at_each_step, axis=1)
        std_dev_prices = np.nanstd(prices_at_each_step, axis=1)

        valid_agg_points = np.isfinite(mean_prices) & np.isfinite(std_dev_prices) & np.isfinite(median_prices)
        if not valid_agg_points.any():
             st.warning("Calculated simulation aggregates contain non-finite values (NaN or Inf) for all steps. This can happen if many simulation paths diverged early.")
             # Ensure all aggregate arrays reflect the non-finite status
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

        # Filter final prices list to only include finite values for summary
        final_prices = [price for price in final_prices_list_raw if np.isfinite(price)]


        # --- Calculate Sidebar Results Here ---
        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        # Ensure last historical price is available and finite (already checked start_price)
        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1]) # Already confirmed finite

        # Ensure final aggregate band values are finite for percentage calculation
        final_upper_price = upper_band[-1] if len(upper_band) > 0 and np.isfinite(upper_band[-1]) else np.nan
        final_lower_price = lower_band[-1] if len(lower_band) > 0 and np.isfinite(lower_band[-1]) else np.nan

        # Percentage Delta to +1 Std Dev
        if np.isfinite(final_upper_price) and last_historical_price_scalar > 0:
             delta_upper_pct = ((final_upper_price - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Percentage Delta to -1 Std Dev
        if np.isfinite(final_lower_price) and last_historical_price_scalar > 0:
             delta_lower_pct = ((final_lower_price - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Risk/Reward Ratio
        if np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
             potential_reward = delta_upper_pct
             potential_risk_abs = -delta_lower_pct # Absolute magnitude of downside movement

             if potential_risk_abs > 0:
                  risk_reward_ratio = potential_reward / potential_risk_abs
             elif potential_risk_abs == 0:
                  risk_reward_ratio = np.inf if potential_reward > 0 else np.nan


    # --- Store Results in Session State ---
    # This makes the results available for the plotting and display functions on the next rerun
    st.session_state.simulation_results = {
        'plot_sim_dates': plot_sim_dates,
        'median_prices': median_prices,
        'mean_prices': mean_prices,
        'std_dev_prices': std_dev_prices,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'final_prices': final_prices, # List of finite final prices
        'simulated_dates': simulated_dates, # Dates for the simulation period
        'delta_upper_pct': delta_upper_pct, # Store sidebar values
        'delta_lower_pct': delta_lower_pct, # Store sidebar values
        'risk_reward_ratio': risk_reward_ratio, # Store sidebar values
        'num_simulations_ran': num_simulations, # Store number of sims run for display
        # Store the raw aggregate arrays if needed for further analysis, but maybe not necessary for plotting/summary
        # 'all_simulated_paths_np': all_simulated_paths_np,
    }
    st.success("Simulation completed and results stored.")
    # Note: Streamlit will automatically rerun the script after this block finishes.


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis, Slider Definition, Historical Analysis Results Display) runs on every rerun.

# --- Display Plot (outside button block) ---
# This runs on every rerun (button click or slider move)
# It gets data from full_historical_data (cached) and reads simulation results from session state.
if full_historical_data is not None and not full_historical_data.empty:
    # Pass simulation results from session state to the plotting function
    fig = plot_simulation(full_historical_data, historical_days_to_display, st.session_state.simulation_results)
    st.pyplot(fig)
    plt.close(fig) # Always close the figure
else:
    # Display a message if full historical data could not be fetched initially
    if 'full_historical_data' in locals() and (full_historical_data is None or full_historical_data.empty):
         st.error("Cannot display plot because historical data fetching failed.")
    else:
         # This case should ideally be caught by the initial check/st.stop(), but defensive
         st.warning("Waiting for historical data...")


# --- Display Sidebar Results (outside button block) ---
# This runs on every rerun. It displays results IF they are in session state.
display_sidebar_results(st.session_state.simulation_results, sidebar_placeholder)

# --- Display Main Results Overview (outside button block) ---
# This runs on every rerun. It displays results IF they are in session state.
display_main_results_overview(st.session_state.simulation_results, historical_data_close_analyzed)
