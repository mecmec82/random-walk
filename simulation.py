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

# --- Explainer Section (Collapsible) ---
with st.expander("📊 How this simulation works"):
    st.write("""
    This app simulates future stock/crypto prices using a **random walk model**, specifically a Geometric Brownian Motion (GBM).
    It's a common model in finance, but it's important to understand its assumptions and limitations.

    **Here's the breakdown:**

    1.  **Data Fetching:** It fetches historical adjusted closing prices for the specified ticker (e.g., SPY, BTC-USD) using the `yfinance` library.
    2.  **Analysis Period:** You select a number of historical trading days (`Historical Trading Days for Analysis`). The app uses the closing prices from *this specific period* to calculate:
        *   The average daily log return (drift).
        *   The **daily volatility** using an **Exponentially Weighted Moving Average (EWMA)** method. EWMA gives more importance to recent price movements based on the `EWMA Decay Factor (Lambda)`. A lambda closer to 1 means recent data are weighted much higher. This volatility is crucial as it determines the *size* of the random steps in the simulation.
    3.  **Random Walk Simulation:** For each simulation step into the future (`Future Simulation Days`), the model calculates the next price based on the previous day's price multiplied by a random factor. This random factor is determined by:
        *   The historical average daily return (the drift).
        *   A random value drawn from a **normal (Gaussian) distribution**.
        *   The size of this random value is scaled by the calculated EWMA daily volatility.

        **New: Start Simulation X Days Ago**: Instead of always starting the random walk from the very last historical data point, you can now choose to start the random walk from 'X' days prior to the last historical data point. This means the simulation will forecast prices starting from a past date, covering the last 'X' days of actual historical movement, and then continue into the future. The actual price path for these 'X' days will be plotted alongside the simulated paths to help you visualize "which path we are currently following" within the forecasted cone of possibilities.
    4.  **Monte Carlo Approach:** The simulation is repeated many times (`Number of Simulations to Run`). Each simulation run generates a different possible price path because of the random component.
    5.  **Aggregation and Results:**
        *   Instead of showing hundreds of individual paths (which would be messy), the app calculates the **median** price across all simulations at each future time step. This gives a sense of the most 'typical' outcome.
        *   It also calculates **standard deviation bands** (+/- 1 and +/- 2 standard deviations) around the mean price at each step. These bands indicate the typical spread of outcomes. Roughly 68% of simulated paths are expected to stay within the +/- 1 Std Dev band, and 95% within the +/- 2 Std Dev band, *if the underlying assumptions held perfectly*.
        *   Key metrics like the expected price movement to the edge of the +/- 1 Std Dev band and a Risk/Reward ratio based on the +/- 1 Std Dev endpoints are calculated from the simulation's final step aggregates.
    6.  **Plotting and Summary:** The historical data (you can adjust how many historical days are *displayed* on the plot using a separate slider, without changing the analysis period) is plotted alongside the simulated median path and the standard deviation bands. A table provides a summary of the historical analysis parameters and the key simulation results.

    **Important Considerations:**

    *   **This is a simplified model:** It assumes future price movements are *random* and follow a normal distribution with constant drift and *predictable* volatility (based on EWMA).
    *   **It does NOT predict the future:** It shows a *range of possible outcomes* based on historical patterns and random chance. Real markets are influenced by news, events, changing fundamentals, and human behavior that are not captured by this simple random walk.
    *   **Volatility is not constant:** While EWMA is better than a simple average, volatility still changes in ways not fully captured (e.g., GARCH models attempt this).
    *   **Returns may not be normally distributed:** Extreme events happen more often than the normal distribution predicts ("fat tails").
    *   Use these results for exploring potential scenarios and risk, not as definitive predictions.
    """)

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
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=10000, step=10)

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
    Includes basic filtering for positive prices and ensures DatetimeIndex.
    """
    #st.info(f"Fetching historical data for {ticker} to cover at least {num_days} days...")

    # Determine start date: go back enough calendar days to cover num_days *trading* days
    # Using 2.5x as a generous estimate for stocks (approx 252 trading days/year)
    # to ensure we fetch enough history even for longer analysis periods.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(num_days * 2.5))


    try:
        # yf.download fetches data between start and end dates.
        # It typically returns a DataFrame with a DatetimeIndex.
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series() # Return empty Series

        # We need the 'Close' price
        historical_data_close = data['Close']

        # Ensure the index is a DatetimeIndex (should be by default from yfinance)
        if not isinstance(historical_data_close.index, pd.DatetimeIndex):
             # Attempt conversion if it's not already
             try:
                  historical_data_close.index = pd.to_datetime(historical_data_close.index)
             except Exception as e:
                  st.error(f"Error converting historical data index to DatetimeIndex: {e}")
                  return pd.Series() # Return empty Series if index is bad

        # Ensure it's sorted by date (ascending)
        historical_data_close = historical_data_close.sort_index()


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


# --- Add Slider for Start Simulation X Days Ago ---
# Max value for X days ago is the number of available trading days in the analysis period minus 1 (to leave at least 1 day for start point)
max_offset_days_allowed = max(0, len(historical_data_close_analyzed) - 1)
start_offset_days = st.sidebar.slider(
    "Start Simulation X Days Ago",
    min_value=0, # 0 means from the last available data point
    value=min(7, max_offset_days_allowed), # Ensure default is within bounds
    max_value=max_offset_days_allowed,
    step=1,
    help="Start the random walk simulation from a historical point, X trading days before the last historical data point used for analysis. This allows you to visualize which simulated paths the actual price has been following during the last X days."
)


# --- Calculate Historical Returns and Volatility (using EWMA) ---
# This calculation happens whenever inputs change, which includes the slider AND EWMA lambda.
# It uses `historical_data_close_analyzed` which is NOT affected by the plot display slider.
# It depends on the cached fetch result, so it's efficient unless ticker, historical_days_requested, or lambda change.
with st.spinner("Calculating historical statistics (including EWMA volatility)..."):
    # Ensure historical_data_close_analyzed is not empty before calculating returns
    if historical_data_close_analyzed.empty:
        st.error("Historical data for analysis is empty. Cannot calculate returns.")
        st.stop()

    log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

    if len(log_returns) < 1: # Need at least one return (2 prices)
        st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after dropping NaNs. Need at least 2 consecutive valid prices.")
        st.stop()

    # Simple Mean Daily Log Return (Still useful as a drift estimate)
    mean_daily_log_return = log_returns.mean()

    # --- Calculate EWMA Daily Log Volatility ---
    # The EWM method uses alpha = 1 - lambda
    ewma_alpha = 1 - ewma_lambda
    ewma_variance_series = None # Initialize in case of error

    try:
        # Calculate EWMA of squared returns (variance)
        # Use min_periods=0 to handle cases with very few returns gracefully (though >=2 are needed)
        # Ensure log_returns is treated as numeric if there's any ambiguity
        ewma_variance_series = log_returns.astype(float).pow(2).ewm(alpha=ewma_alpha, adjust=False, min_periods=0).mean()


        # The EWMA volatility for the *next* step is the square root of the *last* EWMA variance
        # Use .iloc[-1] and .item() to explicitly get the scalar value and avoid FutureWarning
        ewma_daily_log_volatility_scalar = np.sqrt(ewma_variance_series.iloc[-1]).item()

        # Assign to the variable used in simulation
        daily_log_volatility = float(ewma_daily_log_volatility_scalar)


    except Exception as e:
        st.error(f"Error calculating EWMA volatility: {e}")
        daily_log_volatility = np.nan # Indicate failure

    # Type and finiteness checks for calculated values
    # Use .item() for the mean to address FutureWarning
    try:
        # Check if mean_daily_log_return is a pandas Series (it should be a float/numpy scalar from .mean())
        # If it's a Series of length 1 (unexpected), use .item()
        if isinstance(mean_daily_log_return, pd.Series):
             #st.warning("Mean daily log return is a Series, extracting item.")
             mean_float = float(mean_daily_log_return.item())
        else: # Should be a standard float/numpy scalar
            mean_float = float(mean_daily_log_return)

        volatility_float = float(daily_log_volatility) # daily_log_volatility is already scalar float here

    except (TypeError, ValueError) as e:
         st.error(f"Unexpected value or type for calculated mean or volatility after EWMA: {e}")
         st.info(f"Mean value: {mean_daily_log_return}, Mean type: {type(mean_daily_log_return)}")
         st.info(f"Volatility value: {daily_log_volatility}, Volatility type: {type(daily_log_volatility)}")
         st.stop() # Stop before simulation if stats are bad

    if not np.isfinite(mean_float) or not np.isfinite(volatility_float) or volatility_float <= 0:
         st.error(f"Could not calculate finite, positive volatility ({volatility_float:.6f}) or finite mean ({mean_float:.6f}) from historical data. Check data or analysis period.")
         st.stop()

# --- Add Slider for Displayed Historical Days ---
# This slider is defined *outside* the button block so it appears immediately.
# Its max_value is based on the actual number of days successfully fetched and available for analysis.
max_display_days = len(full_historical_data) # Use full_historical_data as max for display slider
# Set default value: requested days, capped by available, min 100 unless less data is available
default_display_days = min(historical_days_requested, max_display_days)
default_display_days = max(100, default_display_days) if max_display_days >= 100 else max_display_days
default_display_days = max(1, default_display_days) # Ensure at least 1 day can be displayed if data is minimal


historical_days_to_display = st.sidebar.slider(
    "Historical Days to Display on Plot",
    min_value=min(1, max_display_days), # Ensure min_value is at least 1 and doesn't exceed available data
    max_value=max_display_days,
    value=default_display_days,
    step=10,
    help="Slide to change the number of historical trading days shown on the chart. Does not affect analysis period or calculated volatility."
)


# --- Helper functions for formatting values safely ---
# Moved outside any conditional blocks to be available on all reruns
def format_value(value, format_str=".2f", default="N/A"):
    """Formats a numerical value, returning default if non-finite."""
    if np.isfinite(value):
        return format(value, format_str)
    return default

def format_percentage(value, format_str=".2f", default="N/A"):
     """Formats a numerical value as a percentage, returning default if non-finite."""
     if np.isfinite(value):
          return f"{format(value, format_str)}%"
     return default


# --- Define the plotting function ---
# This function will be called *outside* the button block to redraw the plot
# It takes the full historical data, the slider value, and the simulation results from session state
def plot_simulation(full_historical_data, historical_days_to_display, simulation_results, ticker, ewma_lambda):
    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Select data for plotting using the slider value and convert to numpy arrays ---
    # Use the full fetched data and take the tail based on the slider value
    historical_data_to_plot = full_historical_data.tail(historical_days_to_display)

    # Convert to numpy arrays explicitly
    historical_dates_to_plot_np = historical_data_to_plot.index.values if not historical_data_to_plot.empty else np.array([])
    historical_prices_to_plot_np = historical_data_to_plot.values if not historical_data_to_plot.empty else np.array([])


    # Plot Historical Data (using the filtered data for display)
    if len(historical_prices_to_plot_np) > 0:
        ax.plot(historical_dates_to_plot_np, historical_prices_to_plot_np, label=f'Historical {ticker} Price ({len(historical_data_to_plot)} days displayed)', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    # Only plot simulation results if they exist in session state
    if simulation_results is not None:
         # Use .get() with default values in case keys are missing due to errors
         plot_sim_dates_pd = simulation_results.get('plot_sim_dates')
         median_prices = simulation_results.get('median_prices')
         mean_prices = simulation_results.get('mean_prices') # Need mean for bands
         std_dev_prices = simulation_results.get('std_dev_prices') # Need std dev for bands
         upper_band = simulation_results.get('upper_band') # This is +1 std dev
         lower_band = simulation_results.get('lower_band') # This is -1 std dev
         upper_band_2std = simulation_results.get('upper_band_2std') # New: +2 std dev
         lower_band_2std = simulation_results.get('lower_band_2std') # New: -2 std dev
         num_simulations_ran = simulation_results.get('num_simulations_ran', 'N/A')
         start_offset_days_used = simulation_results.get('start_offset_days_used', 0) # Get the offset used in simulation
         hist_data_analyzed_for_plot = simulation_results.get('historical_data_close_analyzed') # For plotting green line


         # Convert plot_sim_dates to numpy array
         plot_sim_dates_np = plot_sim_dates_pd.values if plot_sim_dates_pd is not None else np.array([])


         if (len(plot_sim_dates_np) > 0 and
             median_prices is not None and len(median_prices) == len(plot_sim_dates_np) and np.isfinite(median_prices).any() and
             mean_prices is not None and len(mean_prices) == len(plot_sim_dates_np) and
             std_dev_prices is not None and len(std_dev_prices) == len(plot_sim_dates_np) and
             upper_band is not None and len(upper_band) == len(plot_sim_dates_np) and
             lower_band is not None and len(lower_band) == len(plot_sim_dates_np) and
             upper_band_2std is not None and len(upper_band_2std) == len(plot_sim_dates_np) and
             lower_band_2std is not None and len(lower_band_2std) == len(plot_sim_dates_np)):


              # Filter points where median is finite for plotting line and bands
              valid_plot_indices = np.isfinite(median_prices)

              # Apply filter to all related arrays
              plot_sim_dates_valid = plot_sim_dates_np[valid_plot_indices]
              median_prices_valid = median_prices[valid_plot_indices]
              mean_prices_valid = mean_prices[valid_plot_indices]
              std_dev_prices_valid = std_dev_prices[valid_plot_indices]
              upper_band_valid = upper_band[valid_plot_indices]
              lower_band_valid = lower_band[valid_plot_indices]
              upper_band_2std_valid = upper_band_2std[valid_plot_indices]
              lower_band_2std_valid = lower_band_2std[valid_plot_indices]


              if len(plot_sim_dates_valid) > 0:
                 # Plot +/- 2 Std Dev Band FIRST (behind +/- 1)
                 if np.isfinite(upper_band_2std_valid).all() and np.isfinite(lower_band_2std_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_2std_valid, upper_band_2std_valid, color='gold', alpha=0.2, label='+/- 2 Std Dev Band')
                 else:
                      st.warning("+/- 2 Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")

                 # Plot +/- 1 Std Dev Band SECOND
                 if np.isfinite(upper_band_valid).all() and np.isfinite(lower_band_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_valid, upper_band_valid, color='darkorange', alpha=0.3, label='+/- 1 Std Dev Band')
                 else:
                      st.warning("+/- 1 Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")


                 # Plot Median line LAST (on top)
                 ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations_ran} runs)', color='red', linestyle='-', linewidth=2)


                 # Plot the actual historical path during the offset period (green line)
                 if start_offset_days_used > 0 and hist_data_analyzed_for_plot is not None and not hist_data_analyzed_for_plot.empty:
                     # Determine the segment of historical data that overlaps with the simulation forecast
                     sim_start_idx_in_analyzed = len(hist_data_analyzed_for_plot) - 1 - start_offset_days_used

                     if sim_start_idx_in_analyzed >= 0:
                         actual_overlap_data = hist_data_analyzed_for_plot.iloc[sim_start_idx_in_analyzed:]
                         if not actual_overlap_data.empty:
                             ax.plot(actual_overlap_data.index, actual_overlap_data.values,
                                     label=f'Actual Price During Offset ({start_offset_days_used} days)', color='green', linewidth=2, linestyle='--')

                             # Add a vertical dashed line at the point where simulation actually diverges from historical
                             ax.axvline(x=actual_overlap_data.index[-1], color='gray', linestyle=':', linewidth=1, label='Forecast Divergence Point')
                         else:
                             st.warning("Actual overlap data segment is empty, cannot plot green line.")
                     else:
                         st.warning("Calculated historical index for green line is out of bounds, cannot plot green line.")


                 # --- Add Labels at the end ---
                 # Find the last step index where median is finite (use original, unfiltered index)
                 last_valid_step_index_in_simulation = len(median_prices) - 1
                 while last_valid_step_index_in_simulation >= 0 and not np.isfinite(median_prices[last_valid_step_index_in_simulation]):
                      last_valid_step_index_in_simulation -= 1

                 if last_valid_step_index_in_simulation >= 0:
                      # Use the original, unfiltered plot_sim_dates_pd for getting the date object for the text label
                      final_date_text = plot_sim_dates_pd[last_valid_step_index_in_simulation]
                      final_median_price = median_prices[last_valid_step_index_in_simulation]

                      ax.text(final_date_text, final_median_price, f" ${final_median_price:.2f}",
                              color='red', fontsize=10, ha='left', va='center', weight='bold')

                      # Check and add labels for +/- 1 Std Dev band endpoints
                      if np.isfinite(upper_band[last_valid_step_index_in_simulation]):
                           ax.text(final_date_text, upper_band[last_valid_step_index_in_simulation], f" ${upper_band[last_valid_step_index_in_simulation]:.2f}",
                                   color='darkorange', fontsize=9, ha='left', va='bottom')

                      if np.isfinite(lower_band[last_valid_step_index_in_simulation]):
                            ax.text(final_date_text, lower_band[last_valid_step_index_in_simulation], f" ${lower_band[last_valid_step_index_in_simulation]:.2f}",
                                   color='darkorange', fontsize=9, ha='left', va='top')

                      # Check and add labels for +/- 2 Std Dev band endpoints
                      if np.isfinite(upper_band_2std[last_valid_step_index_in_simulation]):
                           ax.text(final_date_text, upper_band_2std[last_valid_step_index_in_simulation], f" ${upper_band_2std[last_valid_step_index_in_simulation]:.2f}",
                                   color='goldenrod', fontsize=9, ha='left', va='bottom')

                      if np.isfinite(lower_band_2std[last_valid_step_index_in_simulation]):
                            ax.text(final_date_text, lower_band_2std[last_valid_step_index_in_simulation], f" ${lower_band_2std[last_valid_step_index_in_simulation]:.2f}",
                                   color='goldenrod', fontsize=9, ha='left', va='top')


                 else:
                     st.warning("Median simulated data contains no finite points, skipping end labels.")

                 ax.legend()
              else:
                   st.warning("No finite aggregate simulation data points available to plot the median line/bands.")


         else:
             st.warning("Could not plot simulation aggregates. Data length issues or all aggregate values are non-finite.")
    # else:
        # st.info("Click 'Run Simulation' to see forecasts.") # Optional message


    # Set plot title to reflect displayed historical range vs analysis range
    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)} days displayed, {len(hist_data_analyzed_for_plot)} days analyzed) and Random Walk Simulation Aggregates (EWMA $\lambda$={ewma_lambda})'
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    return fig


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis Calculation, Slider Definition) runs on every rerun.


# --- Button to Run Simulation ---
# This block runs ONLY when the button is clicked
if st.button("Run Simulation"):
    # Clear previous simulation results from session state
    st.session_state.simulation_results = None

    # --- Calculate Simulation Aggregates (Heavy Computation) ---
    # This happens once per button click
    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} future days (starting {start_offset_days} days ago) using EWMA $\lambda$={ewma_lambda}..."):

        # Calculate effective starting index for simulation within `historical_data_close_analyzed`
        sim_start_idx_in_analyzed = len(historical_data_close_analyzed) - 1 - start_offset_days

        # Error checking for sufficient historical data for the chosen offset
        if sim_start_idx_in_analyzed < 0:
            st.error(f"Cannot run simulation with 'Start Simulation {start_offset_days} Days Ago'. Only {len(historical_data_close_analyzed)} historical days available in analysis period. Please reduce 'X Days Ago' or increase 'Historical Trading Days for Analysis'.")
            st.session_state.simulation_results = None
            st.stop()

        # Get start price and date from the data used for analysis, adjusted by offset
        start_price = float(historical_data_close_analyzed.iloc[sim_start_idx_in_analyzed].item())
        simulation_start_date = historical_data_close_analyzed.index[sim_start_idx_in_analyzed]

        if not np.isfinite(start_price) or start_price <= 0:
             st.error(f"Starting historical price ({start_price}) is not a finite positive number. Cannot start simulation.")
             st.session_state.simulation_results = None
             st.stop()

        # Ensure the pre-calculated mean and volatility are finite and volatility is positive
        loc_sim = mean_float
        scale_sim = volatility_float

        if not np.isfinite(loc_sim) or not np.isfinite(scale_sim) or scale_sim <= 0:
             st.error(f"Calculated historical mean ({loc_sim:.6f}) or EWMA volatility ({scale_sim:.6f}) is not finite or volatility is not positive. Cannot run simulation.")
             st.session_state.simulation_results = None
             st.stop()


        # Determine the total number of steps in the simulated path (from simulation start to forecast end)
        # This includes the 'offset' period (start_offset_days + 1 for the start day itself)
        # and the 'future' period (simulation_days).
        total_forecast_length_in_steps = (start_offset_days + 1) + simulation_days

        # Generate the dates for the entire simulated path (from offset start to forecast end)
        # pd.date_range generates 'periods' dates *including* the start.
        simulated_dates_pd = pd.date_range(start=simulation_start_date, periods=total_forecast_length_in_steps, freq='B')

        if len(simulated_dates_pd) != total_forecast_length_in_steps:
             st.warning(f"Could not generate exactly {total_forecast_length_in_steps} business days starting from {simulation_start_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates_pd)}. Simulation path length will be adjusted.")
        
        sim_path_length = len(simulated_dates_pd) # This is the number of actual points in the series (including the start point)

        if sim_path_length < 2: # Need at least two points to see movement
            st.error(f"Not enough forecast steps generated ({sim_path_length}) for meaningful simulation. Check simulation days or offset.")
            st.session_state.simulation_results = None
            st.stop()

        # The random log returns should be for `sim_path_length - 1` steps (as start_price is the first point)
        all_simulated_log_returns = np.random.normal(
             loc=loc_sim,
             scale=scale_sim,
             size=(num_simulations, sim_path_length - 1)
        )

        start_prices_array = np.full((num_simulations, 1), start_price)
        price_change_factors = np.exp(all_simulated_log_returns)
        cumulative_price_multipliers = np.cumprod(np.concatenate((np.ones((num_simulations, 1)), price_change_factors), axis=1), axis=1)
        all_simulated_paths_np = start_prices_array * cumulative_price_multipliers

        # Ensure shapes match after potential date generation issues
        if all_simulated_paths_np.shape[1] != sim_path_length:
             st.error(f"Mismatch between generated simulation path length ({all_simulated_paths_np.shape[1]}) and expected length ({sim_path_length}). This indicates an internal calculation error. Cannot plot simulation.")
             st.session_state.simulation_results = None
             st.stop()

        # --- Calculate Median, Mean, Standard Deviation Across Simulations ---
        prices_at_each_step = all_simulated_paths_np.T

        median_prices = np.nanmedian(prices_at_each_step, axis=1)
        mean_prices = np.nanmean(prices_at_each_step, axis=1)
        std_dev_prices = np.nanstd(prices_at_each_step, axis=1)

        # Calculate upper/lower bands based on mean +/- std dev
        upper_band = mean_prices + std_dev_prices # +1 Std Dev
        lower_band = mean_prices - std_dev_prices # -1 Std Dev
        upper_band_2std = mean_prices + 2 * std_dev_prices # New: +2 Std Dev
        lower_band_2std = mean_prices - 2 * std_dev_prices # New: -2 Std Dev


        # Filter final prices to exclude non-finite values (robustness)
        final_prices_list_raw = all_simulated_paths_np[:, -1].tolist()
        final_prices = [price for price in final_prices_list_raw if np.isfinite(price)]


        # --- Calculate Forecast Metrics (based on +/- 1 std dev) ---
        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        # Ensure last historical price is available and finite (already checked start_price)
        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1].item())


        # Ensure final aggregate band values are finite for percentage calculation
        final_upper_price_1std = upper_band[-1] if len(upper_band) > 0 and np.isfinite(upper_band[-1]) else np.nan
        final_lower_price_1std = lower_band[-1] if len(lower_band) > 0 and np.isfinite(lower_band[-1]) else np.nan

        # Percentage Delta to +1 Std Dev (from today's last price to forecast end)
        if np.isfinite(final_upper_price_1std) and last_historical_price_scalar > 0:
             delta_upper_pct = ((final_upper_price_1std - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Percentage Delta to -1 Std Dev (from today's last price to forecast end)
        if np.isfinite(final_lower_price_1std) and last_historical_price_scalar > 0:
             delta_lower_pct = ((final_lower_price_1std - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Risk/Reward Ratio (Handle division by zero or negative risk)
        if np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
             potential_reward = delta_upper_pct
             potential_risk_abs = -delta_lower_pct # Absolute magnitude of downside movement

             if potential_risk_abs > 1e-9: # Check if potential risk is meaningfully positive (more than ~0)
                  risk_reward_ratio = potential_reward / potential_risk_abs
             elif potential_risk_abs >= -1e-9 and potential_reward > 1e-9: # Risk is zero or negative (upside), Reward is positive
                  risk_reward_ratio = np.inf
             # If risk is zero/negative and reward is zero/negative, ratio is undefined/N/A


    # --- Store Results in Session State ---
    st.session_state.simulation_results = {
        'historical_data_close_analyzed': historical_data_close_analyzed, # Store this too for the final table & green line
        'mean_float': mean_float,
        'volatility_float': volatility_float,
        'ewma_lambda_used': ewma_lambda,
        'plot_sim_dates': simulated_dates_pd, # This now contains all dates for the forecast range
        'median_prices': median_prices,
        'mean_prices': mean_prices,
        'std_dev_prices': std_dev_prices,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'upper_band_2std': upper_band_2std,
        'lower_band_2std': lower_band_2std,
        'final_prices': final_prices,
        'simulated_dates': simulated_dates_pd, # Dates for the simulation period (including offset)
        'delta_upper_pct': delta_upper_pct,
        'delta_lower_pct': delta_lower_pct,
        'risk_reward_ratio': risk_reward_ratio,
        'num_simulations_ran': num_simulations,
        'simulation_start_date': simulation_start_date, # New: Actual start date of simulation
        'start_offset_days_used': start_offset_days, # New: Offset days used
        'total_sim_steps': total_forecast_length_in_steps, # New: Total steps including offset
        'future_simulation_days_input': simulation_days # New: Original future days input
    }
    #st.success("Simulation completed and results stored.")


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis Calculation, Slider Definition) runs on every rerun.


# --- Display Risk/Reward and Key Forecasts (outside button block) ---
# This runs on every rerun if simulation results are in session state
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results

    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    delta_lower_pct = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio = results.get('risk_reward_ratio', np.nan)
    num_simulations_ran = results.get('num_simulations_ran', 'N/A')


    st.subheader("Simulation Forecast Insights")
    st.write(f"*(Based on {num_simulations_ran} runs)*")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Expected movement to +1 Std Dev End", value=format_percentage(delta_upper_pct, '.2f', 'N/A'))
    with col2:
        st.metric(label="Expected movement to -1 Std Dev End", value=format_percentage(delta_lower_pct, '.2f', 'N/A'))
    with col3:
        risk_reward_str = "N/A"
        if np.isfinite(risk_reward_ratio):
             if risk_reward_ratio == np.inf:
                  risk_reward_str = "Infinite"
             else:
                  risk_reward_str = f"{risk_reward_ratio:.2f} : 1"
        elif np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
             risk_reward_str = "Undetermined / Favorable Downside"

        st.metric(label="Risk/Reward Ratio (+1 Gain : -1 Loss)", value=risk_reward_str)


# --- Display Plot (outside button block) ---
# This runs on every rerun (button click or slider move)
st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")
if full_historical_data is not None and not full_historical_data.empty:
    fig = plot_simulation(
        full_historical_data,
        historical_days_to_display,
        st.session_state.simulation_results,
        ticker,
        ewma_lambda
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    if 'full_historical_data' in locals() and (full_historical_data is None or full_historical_data.empty):
         st.error("Cannot display plot because historical data fetching failed.")
    else:
         st.warning("Waiting for historical data...")


# --- Display Results Table (outside button block, at the bottom) ---
if st.session_state.simulation_results is not None:
    st.subheader("Simulation and Analysis Summary")

    results = st.session_state.simulation_results

    # Get historical stats from stored results
    historical_data_analyzed = results.get('historical_data_close_analyzed')
    hist_analysis_days = len(historical_data_analyzed) if isinstance(historical_data_analyzed, pd.Series) else 0
    hist_mean_log_return = results.get('mean_float', np.nan)
    hist_volatility_log = results.get('volatility_float', np.nan)
    ewma_lambda_used = results.get('ewma_lambda_used', 'N/A')

    # --- SAFELY GET LAST HISTORICAL PRICE AND DATE ---
    last_historical_price_scalar = np.nan
    last_historical_date_analysis = "N/A"

    if isinstance(historical_data_analyzed, pd.Series) and not historical_data_analyzed.empty:
        try:
            raw_last_value = historical_data_analyzed.iloc[-1]
            if np.isfinite(raw_last_value):
                last_historical_price_scalar = float(raw_last_value.item())
                last_historical_date_analysis = historical_data_analyzed.index[-1]
            else:
                 st.warning("Last historical price is non-finite (NaN/Inf), cannot display in table.")
        except Exception as e:
            st.error(f"Error processing last historical price or date for table: {e}")


    # Get simulation results from stored results
    num_sims_ran = results.get('num_simulations_ran', 'N/A')
    sim_dates_pd = results.get('simulated_dates', pd.DatetimeIndex([])) # Get the pandas DatetimeIndex for all simulated steps
    sim_start_date_used = results.get('simulation_start_date', "N/A") # New
    start_offset_days_used = results.get('start_offset_days_used', 0) # New
    total_sim_steps_used = results.get('total_sim_steps', 0) # New
    future_sim_days_input = results.get('future_simulation_days_input', 0) # New

    sim_end_date = sim_dates_pd.max() if len(sim_dates_pd) > 0 else "N/A"


    # Get aggregate results arrays safely
    median_prices_array = results.get('median_prices', np.array([]))
    mean_prices_array = results.get('mean_prices', np.array([]))
    std_dev_prices_array = results.get('std_dev_prices', np.array([]))
    upper_band_array = results.get('upper_band', np.array([]))
    lower_band_array = results.get('lower_band', np.array([]))
    upper_band_2std_array = results.get('upper_band_2std', np.array([]))
    lower_band_2std_array = results.get('lower_band_2std', np.array([]))
    final_prices_list = results.get('final_prices', [])

    # Safely get final values from aggregate arrays
    median_end_price = median_prices_array[-1] if len(median_prices_array) > 0 and np.isfinite(median_prices_array[-1]) else np.nan
    mean_end_price = mean_prices_array[-1] if len(mean_prices_array) > 0 and np.isfinite(mean_prices_array[-1]) else np.nan
    std_dev_end = std_dev_prices_array[-1] if len(std_dev_prices_array) > 0 and np.isfinite(std_dev_prices_array[-1]) else np.nan
    upper_band_end_price_1std = upper_band_array[-1] if len(upper_band_array) > 0 and np.isfinite(upper_band_array[-1]) else np.nan
    lower_band_end_price_1std = lower_band_array[-1] if len(lower_band_array) > 0 and np.isfinite(lower_band_array[-1]) else np.nan
    upper_band_end_price_2std = upper_band_2std_array[-1] if len(upper_band_2std_array) > 0 and np.isfinite(upper_band_2std_array[-1]) else np.nan
    lower_band_end_price_2std = lower_band_2std_array[-1] if len(lower_band_2std_array) > 0 and np.isfinite(lower_band_2std_array[-1]) else np.nan


    # Safely get min/max from the list of *actual* final prices
    actual_min_end_price = np.min(final_prices_list) if final_prices_list else np.nan
    actual_max_end_price = np.max(final_prices_list) if final_prices_list else np.nan

    # Get delta percentages and risk/reward from stored values (calculated based on +/- 1 std dev)
    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    delta_lower_pct = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio_val = results.get('risk_reward_ratio', np.nan)


    # Prepare data for the table
    table_data = {
        'Metric': [
            'Historical Analysis Period (days)',
            'Historical Mean Daily Log Return',
            f'EWMA Daily Log Volatility ($\lambda$={ewma_lambda_used})',
            'Last Historical Price ($)',
            'Last Historical Date',
            '--- Simulation Results ---',
            'Number of Simulations',
            'Simulation Starting X Days Ago',
            'Simulation Start Date',
            'Total Simulation Steps (Offset + Future)',
            'Forecasted Days into Future (from current)',
            'Simulation End Date',
            'Simulated Median Ending Price ($)',
            'Simulated Mean Ending Price ($)',
            'Simulated Std Dev Ending Price ($)',
            'Simulated +1 Std Dev Ending Price ($)',
            'Simulated -1 Std Dev Ending Price ($)',
            'Simulated +2 Std Dev Ending Price ($)',
            'Simulated -2 Std Dev Ending Price ($)',
            'Actual Min Simulated Ending Price ($)',
            'Actual Max Simulated Ending Price ($)',
            'Expected movement to +1 Std Dev End (%)', # Kept consistent with metric
            'Expected movement to -1 Std Dev End (%)', # Kept consistent with metric
            'Risk/Reward Ratio (+1 Gain : -1 Loss)',
        ],
        'Value': [
            hist_analysis_days,
            format_value(hist_mean_log_return, ".6f"),
            format_value(hist_volatility_log, ".6f"),
            format_value(last_historical_price_scalar, ".2f"),
            last_historical_date_analysis.strftime('%Y-%m-%d') if isinstance(last_historical_date_analysis, (datetime, pd.Timestamp)) else str(last_historical_date_analysis),
            '',
            num_sims_ran,
            start_offset_days_used,
            sim_start_date_used.strftime('%Y-%m-%d') if isinstance(sim_start_date_used, (datetime, pd.Timestamp)) else str(sim_start_date_used),
            total_sim_steps_used,
            future_sim_days_input,
            sim_end_date.strftime('%Y-%m-%d') if isinstance(sim_end_date, (datetime, pd.Timestamp)) else str(sim_end_date),
            format_value(median_end_price, ".2f"),
            format_value(mean_end_price, ".2f"),
            format_value(std_dev_end, ".2f"),
            format_value(upper_band_end_price_1std, ".2f"),
            format_value(lower_band_end_price_1std, ".2f"),
            format_value(upper_band_end_price_2std, ".2f"),
            format_value(lower_band_end_price_2std, ".2f"),
            format_value(actual_min_end_price, ".2f"),
            format_value(actual_max_end_price, ".2f"),
            format_percentage(delta_upper_pct, ".2f"),
            format_percentage(delta_lower_pct, ".2f"),
            f"{format_value(risk_reward_ratio_val, '.2f')} : 1" if np.isfinite(risk_reward_ratio_val) and risk_reward_ratio_val != np.inf else ("Infinite" if risk_reward_ratio_val == np.inf else "N/A"),
        ]
    }

    results_df = pd.DataFrame(table_data)
    st.dataframe(results_df, hide_index=True, use_container_width=True)
