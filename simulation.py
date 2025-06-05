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
st.write("Simulate multiple future price movements using random walks based on historical volatility estimated via EWMA, using free data via yfinance.")

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

# EWMA specific parameter
ewma_lambda = st.sidebar.slider(
    "EWMA Decay Factor (Lambda)",
    min_value=0.85,
    max_value=0.99,
    value=0.94, # Common value from RiskMetrics
    step=0.005,
    help="Higher values give more weight to recent returns when calculating volatility."
)


st.sidebar.write("Data fetched using yfinance (Yahoo Finance).")
st.sidebar.write("Volatility estimated using Exponentially Weighted Moving Average (EWMA).")
st.sidebar.write("Note: Yahoo Finance data may have occasional inaccuracies or downtime.")


# --- Helper function to fetch historical data using yfinance ---
# Cached function: runs only when ticker or historical_days_requested changes
@st.cache_data(ttl=3600) # Cache data fetching results for 1 hour (3600 seconds)
def fetch_historical_data(ticker, num_days):
    """
    Fetches historical data for a period covering at least num_days
    using yfinance. Returns a pandas Series of closing prices with datetime index.
    Includes basic filtering for positive prices.
    """
    #st.info(f"Fetching historical data for {ticker} to cover at least {num_days} days...")

    # Determine start date: go back enough calendar days to cover num_days *trading* days
    # Using 2.5x as a generous estimate for stocks (approx 252 trading days/year)
    # to ensure we fetch enough history even for longer analysis periods.
    # Increased buffer slightly for longer historical analysis periods
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(num_days * 2.5))


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
        if historical_data_close.empty:
             st.error("No historical data with positive prices found.")
             return pd.Series()


        # Return the full fetched & filtered Series.
        # We will take the tail for analysis *and* display *after* fetching.
        # This allows the slider to control the display tail without re-fetching the whole period.
        #st.success(f"Successfully fetched {len(historical_data_close)} trading days.")
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
     st.warning("Enter parameters and click 'Run Simulation'. Not enough historical data available for analysis (need at least 2 days with positive prices).")
     # Stop here if data is insufficient, so we don't try to create a slider or calculate stats with invalid range
     st.stop()


# --- Calculate Historical Returns and Volatility (using EWMA) ---
# This calculation happens whenever inputs change, which includes the slider AND EWMA lambda.
# It uses `historical_data_close_analyzed` which is NOT affected by the plot display slider.
# It depends on the cached fetch result, so it's efficient unless ticker, historical_days_requested, or lambda change.
with st.spinner("Calculating historical statistics (including EWMA volatility)..."):
    log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

    if len(log_returns) < 1: # Need at least one return (2 prices)
        st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after dropping NaNs. Need at least 2 consecutive valid prices.")
        st.stop()

    # Simple Mean Daily Log Return (Still useful as a drift estimate)
    mean_daily_log_return = log_returns.mean()

    # --- Calculate EWMA Daily Log Volatility ---
    # The EWM method uses alpha = 1 - lambda
    ewma_alpha = 1 - ewma_lambda
    try:
        # Calculate EWMA of squared returns (variance)
        ewma_variance_series = log_returns.pow(2).ewm(alpha=ewma_alpha, adjust=False).mean()

        # The EWMA volatility for the *next* step is the square root of the *last* EWMA variance
        ewma_daily_log_volatility = np.sqrt(ewma_variance_series.iloc[-1])

        # Assign to the variable used in simulation
        daily_log_volatility = float(ewma_daily_log_volatility) # Ensure it's a standard float


    except Exception as e:
        st.error(f"Error calculating EWMA volatility: {e}")
        daily_log_volatility = np.nan # Indicate failure

    # Type and finiteness checks for calculated values
    try:
        mean_float = float(mean_daily_log_return)
        volatility_float = float(daily_log_volatility)
    except (TypeError, ValueError) as e:
         st.error(f"Unexpected value or type for calculated mean or volatility after EWMA: {e}")
         st.info(f"Mean value: {mean_daily_log_return}, Mean type: {type(mean_daily_log_return)}")
         st.info(f"Volatility value: {daily_log_volatility}, Volatility type: {type(daily_log_volatility)}")
         st.stop() # Stop before simulation if stats are bad

    if not np.isfinite(mean_float) or not np.isfinite(volatility_float) or volatility_float <= 0:
         st.error(f"Could not calculate finite, positive volatility ({volatility_float}) or finite mean ({mean_float}) from historical data. Check data or analysis period.")
         st.stop()


# --- Add Slider for Displayed Historical Days ---
# This slider is defined *outside* the button block so it appears immediately.
# Its max_value is based on the actual number of days successfully fetched and available for analysis.
max_display_days = len(historical_data_close_analyzed)
# Set default value: requested days, capped by available, min 100 unless less data is available
default_display_days = min(historical_days_requested, max_display_days)
default_display_days = max(100, default_display_days) if max_display_days >= 100 else max_display_days


historical_days_to_display = st.sidebar.slider(
    "Historical Days to Display on Plot",
    min_value=min(100, max_display_days) if max_display_days >= 100 else max_display_days, # Ensure min_value doesn't exceed available data
    max_value=max_display_days,
    value=default_display_days,
    step=10,
    help="Slide to change the number of historical trading days shown on the chart. Does not affect analysis period or calculated volatility."
)


# --- Sidebar Results Display Area ---
st.sidebar.header("Simulation Insights")
sidebar_placeholder = st.sidebar.empty() # Create a placeholder to update results



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
        # Use .values to plot the numpy array, ensuring Matplotlib gets numbers directly
        ax.plot(historical_dates_to_plot, historical_data_to_plot.values, label=f'Historical {ticker} Price ({len(historical_data_to_plot)} days displayed)', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    # Only plot simulation results if they exist in session state
    if simulation_results is not None:
         # Unpack results from the session state dictionary
         plot_sim_dates = simulation_results.get('plot_sim_dates')
         median_prices = simulation_results.get('median_prices')
         upper_band = simulation_results.get('upper_band')
         lower_band = simulation_results.get('lower_band')
         num_simulations_ran = simulation_results.get('num_simulations_ran', 'N/A') # Get the actual number of sims ran

         # Check if plotting aggregates is possible based on the simulation results
         # Need dates, median, upper, and lower bands of consistent length and not empty
         if (plot_sim_dates is not None and len(plot_sim_dates) > 0 and
             median_prices is not None and len(median_prices) == len(plot_sim_dates) and np.isfinite(median_prices).any() and
             upper_band is not None and len(upper_band) == len(plot_sim_dates) and
             lower_band is not None and len(lower_band) == len(plot_sim_dates)):


              # Filter points where median is finite for plotting line and band
              valid_plot_indices = np.isfinite(median_prices)
              plot_sim_dates_valid = plot_sim_dates[valid_plot_indices]
              median_prices_valid = median_prices[valid_plot_indices]
              upper_band_valid = upper_band[valid_plot_indices]
              lower_band_valid = lower_band[valid_plot_indices]


              if len(plot_sim_dates_valid) > 0:
                 ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations_ran} runs)', color='red', linestyle='-', linewidth=2)

                 # Ensure bands also have valid finite points corresponding to the valid median points
                 if np.isfinite(upper_band_valid).all() and np.isfinite(lower_band_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_valid, upper_band_valid, color='orange', alpha=0.3, label='+/- 1 Std Dev Band')
                 else:
                      st.warning("Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")
                      # Attempt to plot if there's at least one valid segment in the band, but this gets complicated fast
                      # For simplicity, we require all points corresponding to the median's valid points to be finite for the band plot


                 # --- Add Labels at the end ---
                 # Find the last step index where median is finite
                 last_valid_step_index_in_simulation = len(median_prices) - 1
                 while last_valid_step_index_in_simulation >= 0 and not np.isfinite(median_prices[last_valid_step_index_in_simulation]):
                      last_valid_step_index_in_simulation -= 1

                 if last_valid_step_index_in_simulation >= 0:
                      final_date = plot_sim_dates[last_valid_step_index_in_simulation]
                      final_median_price = median_prices[last_valid_step_index_in_simulation]

                      ax.text(final_date, final_median_price, f" ${final_median_price:.2f}",
                              color='red', fontsize=10, ha='left', va='center', weight='bold')

                      # Check if the band values are finite at this specific last valid step
                      if np.isfinite(upper_band[last_valid_step_index_in_simulation]):
                           ax.text(final_date, upper_band[last_valid_step_index_in_simulation], f" ${upper_band[last_valid_step_index_in_simulation]:.2f}",
                                   color='darkorange', fontsize=9, ha='left', va='bottom')

                      if np.isfinite(lower_band[last_valid_step_index_in_simulation]):
                            ax.text(final_date, lower_band[last_valid_step_index_in_simulation], f" ${lower_band[last_valid_step_index_in_simulation]:.2f}",
                                   color='darkorange', fontsize=9, ha='left', va='top')

                 else:
                     st.warning("Median simulated data contains no finite points, skipping end labels.")

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
    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)} days displayed, {len(historical_data_close_analyzed)} days analyzed) and Random Walk Simulation Aggregates (EWMA $\lambda$={ewma_lambda})'
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: format dates
    #fig.autofmt_xdate() # Optional: tilt dates for readability

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
              # Use .get() with default values in case keys are missing due to errors
              delta_upper_pct = simulation_results.get('delta_upper_pct', np.nan)
              delta_lower_pct = simulation_results.get('delta_lower_pct', np.nan)
              risk_reward_ratio = simulation_results.get('risk_reward_ratio', np.nan)
              num_simulations_ran = simulation_results.get('num_simulations_ran', 'N/A')

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
                   st.write("Ratio (+1 Gain : -1 Loss): **Undetermined / Favorable Downside**") # Upper finite, Lower is 0 or positive
              else:
                   st.write("Ratio (+1 Gain : -1 Loss): **N/A**")

              st.write(f"*(Based on {num_simulations_ran} runs)*")
         else:
              # Display initial message if no simulation run yet
              st.info("Click 'Run Simulation' to see forecasts.")


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis Calculation, Slider Definition, Historical Analysis Results Display) runs on every rerun.

# --- Button to Run Simulation ---
# This block runs ONLY when the button is clicked
if st.button("Run Simulation"):
    # Clear previous sidebar results visual (placeholder)
    sidebar_placeholder.empty()
    # Clear previous simulation results from session state
    st.session_state.simulation_results = None # Clear old results

    # --- Calculate Simulation Aggregates (Heavy Computation) ---
    # This happens once per button click
    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days using EWMA $\lambda$={ewma_lambda}..."):

        # Get start price from the data used for analysis
        try:
            raw_start_price_value = historical_data_close_analyzed.iloc[-1]
            start_price = float(raw_start_price_value)
        except (TypeError, ValueError) as e:
             st.error(f"Unexpected value or type for last historical price before conversion: {e}")
             start_price = np.nan # Set to NaN if conversion fails

        if not np.isfinite(start_price) or start_price <= 0:
             st.error(f"Last historical price ({start_price}) is not a finite positive number after conversion. Cannot start simulation.")
             # Set session state to None to indicate simulation failed/skipped
             st.session_state.simulation_results = None
             st.stop()

        # Ensure the pre-calculated mean and volatility are finite and volatility is positive
        # These checks are already done above, but reinforce before using them
        loc_sim = mean_float
        scale_sim = volatility_float

        if not np.isfinite(loc_sim) or not np.isfinite(scale_sim) or scale_sim <= 0:
             st.error(f"Calculated historical mean ({loc_sim:.6f}) or EWMA volatility ({scale_sim:.6f}) is not finite or volatility is not positive. Cannot run simulation.")
             st.session_state.simulation_results = None
             st.stop()


        # Prepare dates for simulation results plotting (based on analysis dates)
        historical_dates_analysis = historical_data_close_analyzed.index
        last_historical_date_analysis = historical_dates_analysis.max()

        simulated_dates = pd.DatetimeIndex([])
        sim_path_length = 0

        try:
            # Generate future dates, handling potential errors
            # Increased periods by 1 to include the day *after* the last historical day as the first simulation day
            simulated_dates = pd.date_range(start=last_historical_date_analysis, periods=simulation_days + 1, freq='B')[1:] # [1:] excludes start date
            if len(simulated_dates) != simulation_days:
                 st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date_analysis.strftime('%Y-%m-%d')}. Generated {len(simulated_dates)}. Simulation path length will be adjusted.")
            sim_path_length = len(simulated_dates) + 1 # Add 1 for the starting price point
        except Exception as date_range_error:
             st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
             st.session_state.simulation_results = None
             st.stop()

        # Prepare the full date axis for plotting (Historical End + Simulated Dates)
        plot_sim_dates = pd.DatetimeIndex([])
        if len(simulated_dates) > 0 and sim_path_length > 0:
            # Combine the last historical date with the simulated future dates
            last_historical_date_index = pd.DatetimeIndex([last_historical_date_analysis])
            plot_sim_dates = last_historical_date_index.append(simulated_dates)

            if len(plot_sim_dates) != sim_path_length:
                 st.error(f"Mismatch between calculated path length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates)}). Cannot plot.")
                 st.session_state.simulation_results = None
                 st.stop()
        else:
             st.error("Skipping simulation as future dates could not be generated or have zero length.")
             st.session_state.simulation_results = None
             st.stop()


        # --- Run Multiple Simulations (Inner loop) ---
        # Use the calculated EWMA volatility (scale_sim)
        # Generate all random log returns for all simulations at once for efficiency
        all_simulated_log_returns = np.random.normal(
             loc=loc_sim,    # Mean drift
             scale=scale_sim, # EWMA Volatility
             size=(num_simulations, simulation_days) # Shape is (num_simulations, num_steps)
        )

        # Calculate simulated price paths efficiently using cumulative product
        # Start with an array of starting prices for each simulation
        start_prices_array = np.full((num_simulations, 1), start_price)
        # Convert log returns to price change factors (1 + percentage change)
        price_change_factors = np.exp(all_simulated_log_returns)
        # Calculate cumulative product of price changes, starting with 1 (no change from start)
        cumulative_price_multipliers = np.cumprod(np.concatenate((np.ones((num_simulations, 1)), price_change_factors), axis=1), axis=1)
        # Multiply the starting price by the cumulative multipliers to get the price paths
        all_simulated_paths_np = start_prices_array * cumulative_price_multipliers

        # Ensure the shape matches sim_path_length (handle potential date trimming)
        # This shouldn't happen if date generation was successful and matched simulation_days
        if all_simulated_paths_np.shape[1] > sim_path_length:
             st.warning(f"Simulation paths calculated with length {all_simulated_paths_np.shape[1]}, but expected length is {sim_path_length} due to date generation. Truncating paths.")
             all_simulated_paths_np = all_simulated_paths_np[:, :sim_path_length]
             # Recalculate final prices from the truncated paths
             final_prices_list_raw = all_simulated_paths_np[:, -1].tolist()
        else:
             final_prices_list_raw = all_simulated_paths_np[:, -1].tolist()


        # --- Calculate Median, Mean, Standard Deviation Across Simulations ---
        # Transpose the array so each row is a timestep, and columns are simulations
        prices_at_each_step = all_simulated_paths_np.T # Shape (sim_path_length, num_simulations)

        # Use nan functions in case any paths resulted in NaN/Inf (shouldn't happen with finite start/mean/vol)
        median_prices = np.nanmedian(prices_at_each_step, axis=1)
        mean_prices = np.nanmean(prices_at_each_step, axis=1)
        std_dev_prices = np.nanstd(prices_at_each_step, axis=1)

        # Calculate upper/lower bands based on mean +/- std dev
        upper_band = mean_prices + std_dev_prices
        lower_band = mean_prices - std_dev_prices

        # Filter final prices to exclude non-finite values (robustness)
        final_prices = [price for price in final_prices_list_raw if np.isfinite(price)]


        # --- Calculate Sidebar Results Here ---
        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        # Ensure last historical price is available and finite (already checked start_price)
        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1]) # Already confirmed finite and positive

        # Ensure final aggregate band values are finite for percentage calculation
        final_upper_price = upper_band[-1] if len(upper_band) > 0 and np.isfinite(upper_band[-1]) else np.nan
        final_lower_price = lower_band[-1] if len(lower_band) > 0 and np.isfinite(lower_band[-1]) else np.nan

        # Percentage Delta to +1 Std Dev
        if np.isfinite(final_upper_price) and last_historical_price_scalar > 0:
             delta_upper_pct = ((final_upper_price - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Percentage Delta to -1 Std Dev
        if np.isfinite(final_lower_price) and last_historical_price_scalar > 0:
             delta_lower_pct = ((final_lower_price - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Risk/Reward Ratio (Handle division by zero or negative risk)
        if np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
             potential_reward = delta_upper_pct
             potential_risk_abs = -delta_lower_pct # Absolute magnitude of downside movement

             if potential_risk_abs > 1e-9: # Check if potential risk is meaningfully positive
                  risk_reward_ratio = potential_reward / potential_risk_abs
             elif potential_risk_abs <= 1e-9 and potential_reward > 0: # Risk is zero or negative (upside), Reward is positive
                  risk_reward_ratio = np.inf
             # If risk is zero/negative and reward is zero/negative, ratio is undefined/N/A


    # --- Store Results in Session State ---
    # This makes the results available for the plotting and display functions on the next rerun
    # Store all necessary variables, including the original data used for analysis summary
    st.session_state.simulation_results = {
        'historical_data_close_analyzed': historical_data_close_analyzed, # Store this too for the final table
        'mean_float': mean_float, # Store historical stats
        'volatility_float': volatility_float, # Store historical stats (EWMA)
        'ewma_lambda_used': ewma_lambda, # Store lambda used
        'plot_sim_dates': plot_sim_dates,
        'median_prices': median_prices,
        'mean_prices': mean_prices, # Store aggregate mean for the table
        'std_dev_prices': std_dev_prices, # Store aggregate std dev for the table
        'upper_band': upper_band,
        'lower_band': lower_band,
        'final_prices': final_prices, # List of finite actual final prices
        'simulated_dates': simulated_dates, # Dates for the simulation period (without start)
        'delta_upper_pct': delta_upper_pct, # Store sidebar values
        'delta_lower_pct': delta_lower_pct, # Store sidebar values
        'risk_reward_ratio': risk_reward_ratio, # Store sidebar values
        'num_simulations_ran': num_simulations, # Store number of sims run for display
    }
    #st.success("Simulation completed and results stored.")
    # Note: Streamlit will automatically rerun the script after this block finishes.


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis Calculation, Slider Definition, Historical Analysis Results Display) runs on every rerun.

# --- Display Plot (outside button block) ---
# This runs on every rerun (button click or slider move)
# It gets data from full_historical_data (cached) and reads simulation results from session state.
st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")
if full_historical_data is not None and not full_historical_data.empty:
    # Pass the full fetched data and simulation results from session state to the plotting function
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


# --- Display Results Table (outside button block, at the bottom) ---
# This runs on every rerun. It displays the table IF simulation results are in session state.
if st.session_state.simulation_results is not None:
    st.subheader("Simulation and Analysis Summary")

    results = st.session_state.simulation_results

    # Get historical stats from stored results
    hist_analysis_days = len(results.get('historical_data_close_analyzed', []))
    hist_mean_log_return = results.get('mean_float', np.nan)
    hist_volatility_log = results.get('volatility_float', np.nan) # This is the EWMA volatility
    ewma_lambda_used = results.get('ewma_lambda_used', 'N/A')

    # Get last historical price safely
    historical_data_analyzed = results.get('historical_data_close_analyzed')
    last_historical_price_scalar = float(historical_data_analyzed.iloc[-1]) if historical_data_analyzed is not None and not historical_data_analyzed.empty and np.isfinite(historical_data_analyzed.iloc[-1]) else np.nan
    last_historical_date_analysis = historical_data_analyzed.index.max() if historical_data_analyzed is not None and not historical_data_analyzed.empty else "N/A"


    # Get simulation results from stored results
    num_sims_ran = results.get('num_simulations_ran', 'N/A')
    sim_dates_list = results.get('simulated_dates', [])
    sim_days = len(sim_dates_list) # Number of steps into the future
    sim_end_date = sim_dates_list.max() if len(sim_dates_list) > 0 else "N/A"

    median_prices_array = results.get('median_prices', np.array([]))
    mean_prices_array = results.get('mean_prices', np.array([]))
    std_dev_prices_array = results.get('std_dev_prices', np.array([]))
    upper_band_array = results.get('upper_band', np.array([]))
    lower_band_array = results.get('lower_band', np.array([]))
    final_prices_list = results.get('final_prices', []) # List of finite actual final prices

    # Safely get final values from arrays
    median_end_price = median_prices_array[-1] if len(median_prices_array) > 0 and np.isfinite(median_prices_array[-1]) else np.nan
    mean_end_price = mean_prices_array[-1] if len(mean_prices_array) > 0 and np.isfinite(mean_prices_array[-1]) else np.nan
    std_dev_end = std_dev_prices_array[-1] if len(std_dev_prices_array) > 0 and np.isfinite(std_dev_prices_array[-1]) else np.nan
    upper_band_end_price = upper_band_array[-1] if len(upper_band_array) > 0 and np.isfinite(upper_band_array[-1]) else np.nan
    lower_band_end_price = lower_band_array[-1] if len(lower_band_array) > 0 and np.isfinite(lower_band_array[-1]) else np.nan

    actual_min_end_price = np.min(final_prices_list) if final_prices_list else np.nan
    actual_max_end_price = np.max(final_prices_list) if final_prices_list else np.nan

    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    delta_lower_pct = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio_val = results.get('risk_reward_ratio', np.nan)


    # Prepare data for the table
    table_data = {
        'Metric': [
            'Historical Analysis Period (days)',
            'Historical Mean Daily Log Return',
            f'EWMA Daily Log Volatility ($\lambda$={ewma_lambda_used})', # Update text
            '--- Simulation Results ---',
            'Number of Simulations',
            f'Simulation Period (days from {(last_historical_date_analysis.strftime("%Y-%m-%d") if isinstance(last_historical_date_analysis, datetime) else str(last_historical_date_analysis))})',
            f'Simulation End Date',
            'Simulated Median Ending Price ($)',
            'Simulated Mean Ending Price ($)',
            'Simulated Std Dev Ending Price ($)',
            'Simulated +1 Std Dev Ending Price ($)',
            'Simulated -1 Std Dev Ending Price ($)',
            'Actual Min Simulated Ending Price ($)',
            'Actual Max Simulated Ending Price ($)',
            'Expected movement to +1 Std Dev End (%)',
            'Expected movement to -1 Std Dev End (%)',
            'Risk/Reward Ratio (+1 Gain : -1 Loss)',
        ],
        'Value': [
            hist_analysis_days,
            f"{hist_mean_log_return:.6f}" if np.isfinite(hist_mean_log_return) else "N/A",
            f"{hist_volatility_log:.6f}" if np.isfinite(hist_volatility_log) else "N/A",
            '', # Separator
            num_sims_ran,
            sim_days,
            sim_end_date.strftime('%Y-%m-%d') if isinstance(sim_end_date, datetime) else str(sim_end_date),
            f"{median_end_price:.2f}" if np.isfinite(median_end_price) else "N/A",
            f"{mean_end_price:.2f}" if np.isfinite(mean_end_price) else "N/A",
            f"{std_dev_end:.2f}" if np.isfinite(std_dev_end) else "N/A",
            f"{upper_band_end_price:.2f}" if np.isfinite(upper_band_end_price) else "N/A",
            f"{lower_band_end_price:.2f}" if np.isfinite(lower_band_end_price) else "N/A",
            f"{actual_min_end_price:.2f}" if np.isfinite(actual_min_end_price) else "N/A",
            f"{actual_max_end_price:.2f}" if np.isfinite(actual_max_end_price) else "N/A",
            f"{delta_upper_pct:.2f}%" if np.isfinite(delta_upper_pct) else "N/A",
            f"{delta_lower_pct:.2f}%" if np.isfinite(delta_lower_pct) else "N/A",
            f"{risk_reward_ratio_val:.2f} : 1" if np.isfinite(risk_reward_ratio_val) and risk_reward_ratio_val != np.inf else ("Infinite" if risk_reward_ratio_val == np.inf else "N/A"),
        ]
    }

    # Create DataFrame and display
    results_df = pd.DataFrame(table_data)
    st.dataframe(results_df, hide_index=True, use_container_width=True)
