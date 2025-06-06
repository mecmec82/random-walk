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
        *   The calculated mean drift (potentially biased).
        *   A random value drawn from a **normal (Gaussian) distribution**.
        *   The size of this random value is scaled by the calculated EWMA daily volatility.
    4.  **Monte Carlo Approach:** The simulation is repeated many times (`Number of Simulations to Run`). Each simulation run generates a different possible price path because of the random component.
    5.  **Aggregation and Results:**
        *   Instead of showing hundreds of individual paths (which would be messy), the app calculates the **median** price across all simulations at each future time step. This gives a sense of the most 'typical' outcome.
        *   It also calculates **standard deviation bands** (+/- 1 and +/- 2 standard deviations) around the mean price at each step. These bands indicate the typical spread of outcomes. Roughly 68% of simulated paths are expected to stay within the +/- 1 Std Dev band, and 95% within the +/- 2 Std Dev band, *if the underlying assumptions held perfectly*.
        *   Key metrics like the expected price movement to the edge of the +/- 1 Std Dev band and a Risk/Reward ratio based on the +/- 1 Std Dev endpoints are calculated from the simulation's final step aggregates.
    6.  **Plotting and Summary:** The historical data (you can adjust how many historical days are *displayed* on the plot using a separate slider, without changing the analysis period) is plotted alongside the simulated median path and the standard deviation bands. A table provides a summary of the historical analysis parameters and the key simulation results.

    **New Mean Bias Feature (Optional):**

    *   You can introduce a bias to the calculated historical mean daily return based on whether the last price is above or below a Simple Moving Average (SMA).
    *   If the last price is above the SMA (over the `SMA Period for Bias`), the mean drift used in the simulation is slightly increased. If it's below the SMA, the mean drift is slightly decreased.
    *   The magnitude of the bias is controlled by the `Bias Multiplier` multiplied by the absolute value of the original historical mean return. A multiplier of 0 means no bias is applied.
    *   This is a simple form of incorporating a basic trend-following idea into the drift, but it's still a simplification and doesn't capture complex market dynamics.

    **Important Considerations:**

    *   **This is a simplified model:** It assumes future price movements are *random* and follow a normal distribution with constant drift and *predictable* volatility (based on EWMA), plus a simple SMA-based directional bias on the mean.
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

# --- SMA Bias Settings ---
st.sidebar.header("Mean Bias (SMA)")
sma_period = st.sidebar.number_input("SMA Period for Bias", min_value=10, value=100, step=10, help="Period for the Simple Moving Average used to bias the mean drift.")
bias_multiplier = st.sidebar.number_input("Bias Multiplier", min_value=0.0, value=0.5, step=0.05, format="%.2f", help="Multiplier for biasing the mean drift. If price is above SMA, mean increases by this factor * abs(mean). If below, decreases.")
st.sidebar.markdown(f"*(Bias amount = `{bias_multiplier} * |Historical Mean|`)*")


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
    # Add buffer for SMA calculation as well
    end_date = datetime.now()
    # Need historical_days_requested + sma_period data for SMA calculation
    # Total calendar days needed = (historical_days_requested + sma_period) * buffer
    days_buffer = max(historical_days_requested, sma_period) # Ensure we fetch enough for the larger period
    start_date = end_date - timedelta(days=int(days_buffer * 2.5))


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
# It fetches enough data to satisfy the historical_days_requested plus the SMA period buffer.
full_historical_data = fetch_historical_data(
    ticker=ticker,
    # Fetch data for analysis period PLUS SMA period
    num_days=historical_days_requested + sma_period # Increased fetch range
)

# Select the specific subset for ANALYSIS (the most recent historical_days_requested days)
# This is the data used downstream for calculating mean/volatility and starting the simulation.
# This calculation runs on every rerun, but uses the cached full_historical_data.
# We need enough data *before* the analysis period starts to calculate the SMA at the start of the analysis period.
# So we take the tail covering analysis period + SMA period.
historical_data_for_sma_and_analysis = full_historical_data.tail(historical_days_requested + sma_period)


# --- Check if we have enough data for analysis and SMA ---
# This check runs on initial load and input changes (before button click)
if historical_data_for_sma_and_analysis.empty or len(historical_data_for_sma_and_analysis) < historical_days_requested + sma_period:
     st.warning(f"Enter parameters and click 'Run Simulation'. Not enough historical data available for analysis ({len(historical_data_for_sma_and_analysis)} days). Need at least {historical_days_requested + sma_period} days for EWMA and {sma_period}-day SMA calculation.")
     st.stop()


# --- Calculate Historical Returns and Volatility (using EWMA) and Mean (with SMA bias) ---
# This calculation happens whenever inputs change.
# It uses `historical_data_for_sma_and_analysis`
with st.spinner("Calculating historical statistics (EWMA volatility & SMA bias)..."):

    # The data needed for EWMA volatility is log returns *over the analysis period*.
    # We apply the SMA bias based on the price/SMA at the *end* of the analysis period.

    # First, isolate the data strictly for the analysis period (excluding the extra days needed for SMA calculation)
    # This is the data `historical_days_requested` long.
    historical_data_close_analyzed = historical_data_for_sma_and_analysis.tail(historical_days_requested)

    if historical_data_close_analyzed.empty:
        st.error("Historical data for analysis period is empty after selecting tail.")
        st.stop()

    # Calculate log returns for the analysis period
    log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

    if len(log_returns) < 1: # Need at least one return (2 prices in analyzed period)
        st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices in analysis period) to calculate returns and volatility after dropping NaNs. Need at least 2 consecutive valid prices.")
        st.stop()

    # Simple Mean Daily Log Return (This is the original mean before bias)
    mean_daily_log_return = log_returns.mean()

    # --- Calculate EWMA Daily Log Volatility ---
    # Calculated on returns over the analysis period
    ewma_alpha = 1 - ewma_lambda
    try:
        ewma_variance_series = log_returns.astype(float).pow(2).ewm(alpha=ewma_alpha, adjust=False, min_periods=0).mean()
        ewma_daily_log_volatility_scalar = np.sqrt(ewma_variance_series.iloc[-1]).item()
        daily_log_volatility = float(ewma_daily_log_volatility_scalar)
    except Exception as e:
        st.error(f"Error calculating EWMA volatility: {e}")
        daily_log_volatility = np.nan # Indicate failure

    # --- Calculate SMA and Apply Bias to Mean ---
    original_mean_float = float(mean_daily_log_return.item()) # Store the original mean
    biased_mean_float = original_mean_float   # Start with the original mean


    # Calculate the SMA over the larger data window
    sma_series = historical_data_for_sma_and_analysis.rolling(window=sma_period).mean().dropna()

    # Ensure SMA series has a value corresponding to the end of the analysis period
    # The index of sma_series aligns with the end date of the window.
    # The last value of sma_series should correspond to the SMA ending on the last day of
    # the `historical_data_for_sma_and_analysis` series.
    # The price to compare is the last price in the `historical_data_close_analyzed` series.

    if not sma_series.empty and sma_series.index.max() >= historical_data_close_analyzed.index.max():
         try:
              # Get the last price from the data used for ANALYSIS
              last_price_for_sma_comp = historical_data_close_analyzed.iloc[-1].item()

              # Get the SMA value corresponding to the last date in the analysis period
              # Use .loc to align by date, .iloc[-1] on the result just in case of multiple entries for a date (unlikely with daily)
              last_sma_value = sma_series.loc[historical_data_close_analyzed.index.max()].iloc[-1].item()


              st.sidebar.write(f"Last Price vs {sma_period}-Day SMA:")
              st.sidebar.write(f"  Last Price ({historical_data_close_analyzed.index.max().strftime('%Y-%m-%d')}): ${last_price_for_sma_comp:.2f}")
              st.sidebar.write(f"  {sma_period}-Day SMA ({sma_series.index.max().strftime('%Y-%m-%d')}): ${last_sma_value:.2f}")


              bias_amount = bias_multiplier * abs(original_mean_float) # Calculate bias amount

              if last_price_for_sma_comp > last_sma_value:
                  biased_mean_float = original_mean_float + bias_amount
                  st.sidebar.info(f"Last price ABOVE SMA. Applying UPWARD bias ({bias_amount:.6f}) to mean drift.")
              elif last_price_for_sma_comp < last_sma_value:
                  biased_mean_float = original_mean_float - bias_amount
                  st.sidebar.info(f"Last price BELOW SMA. Applying DOWNWARD bias (-{bias_amount:.6f}) to mean drift.")
              else: # last_price_for_sma_comp == last_sma_value
                  biased_mean_float = original_mean_float
                  st.sidebar.info(f"Last price EQUAL to SMA. No bias applied to mean drift.")

              st.sidebar.write(f"  Original Mean Drift: {original_mean_float:.6f}")
              st.sidebar.write(f"  Biased Mean Drift: {biased_mean_float:.6f}")

         except Exception as e:
              st.error(f"Error calculating or applying SMA bias based on specific date: {e}")
              st.sidebar.error("SMA bias calculation failed, using original mean.")
              biased_mean_float = original_mean_float # Ensure it's the original mean if bias fails
    else:
        st.warning(f"Not enough historical data for {sma_period}-day SMA calculation to cover the analysis period end date. Need data going back at least {historical_days_requested + sma_period} days.")
        st.sidebar.warning("SMA bias calculation skipped, using original mean.")
        biased_mean_float = original_mean_float # Ensure it's the original mean if bias fails


    # Now, use the potentially biased mean for the simulation
    loc_sim = biased_mean_float
    scale_sim = volatility_float # Volatility calculation remains the same

    # --- Type and finiteness checks for calculated values ---
    # Checks for mean_float and volatility_float are done on the values *used in the simulation*
    try:
        loc_sim_float_checked = float(loc_sim) # Ensure loc_sim is convertible to float
        scale_sim_float_checked = float(scale_sim) # Ensure scale_sim is convertible to float
    except (TypeError, ValueError) as e:
         st.error(f"Final calculated mean or volatility has unexpected type/value before simulation: {e}")
         st.info(f"Mean value: {loc_sim}, Volatility value: {scale_sim}")
         st.stop() # Stop before simulation if stats are bad

    if not np.isfinite(loc_sim_float_checked) or not np.isfinite(scale_sim_float_checked) or scale_sim_float_checked <= 0:
         st.error(f"Final calculated mean ({loc_sim_float_checked:.6f}) or volatility ({scale_sim_float_checked:.6f}) is not finite or volatility is not positive. Cannot run simulation.")
         st.stop()

    # Store the validated float values to be used in the simulation
    mean_float_for_sim = loc_sim_float_checked
    volatility_float_for_sim = scale_sim_float_checked


# --- Add Slider for Displayed Historical Days ---
# This slider is defined *outside* the button block so it appears immediately.
# Its max_value is based on the actual number of days successfully fetched and available for analysis.
# max_display_days should be based on the data used for analysis (historical_days_requested)
max_display_days = len(historical_data_close_analyzed) # Use the length of the *analysis period* data
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
    help="Slide to change the number of historical trading days shown on the chart. Does not affect analysis period or calculated volatility/bias."
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
def plot_simulation(full_historical_data, historical_days_to_display, simulation_results, ticker, historical_data_close_analyzed_len, ewma_lambda, sma_period, bias_multiplier):
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
         num_simulations_ran = simulation_results.get('num_simulations_ran', 'N/A') # Get the actual number of sims ran

         # Check if plotting aggregates is possible based on the simulation results
         # Need dates, median, mean, std_dev and bands of consistent length and not empty
         # Convert plot_sim_dates to numpy array
         plot_sim_dates_np = plot_sim_dates_pd.values if plot_sim_dates_pd is not None else np.array([])


         if (len(plot_sim_dates_np) > 0 and
             median_prices is not None and len(median_prices) == len(plot_sim_dates_np) and np.isfinite(median_prices).any() and
             mean_prices is not None and len(mean_prices) == len(plot_sim_dates_np) and
             std_dev_prices is not None and len(std_dev_prices) == len(plot_sim_dates_np) and
             upper_band is not None and len(upper_band) == len(plot_sim_dates_np) and
             lower_band is not None and len(lower_band) == len(plot_sim_dates_np) and
             upper_band_2std is not None and len(upper_band_2std) == len(plot_sim_dates_np) and # Check new bands
             lower_band_2std is not None and len(lower_band_2std) == len(plot_sim_dates_np)):


              # Filter points where median is finite for plotting line and bands
              # Note: The validity check here is primarily on the median for drawing the lines/bands based on it
              valid_plot_indices = np.isfinite(median_prices)

              # Apply filter to all related arrays
              plot_sim_dates_valid = plot_sim_dates_np[valid_plot_indices]
              median_prices_valid = median_prices[valid_plot_indices]
              mean_prices_valid = mean_prices[valid_plot_indices]
              std_dev_prices_valid = std_dev_prices[valid_plot_indices]
              upper_band_valid = upper_band[valid_plot_indices]
              lower_band_valid = lower_band[valid_plot_indices]
              upper_band_2std_valid = upper_band_2std[valid_plot_indices] # Filter new bands
              lower_band_2std_valid = lower_band_2std[valid_plot_indices] # Filter new bands


              if len(plot_sim_dates_valid) > 0:
                 # Plot +/- 2 Std Dev Band FIRST (behind +/- 1)
                 if np.isfinite(upper_band_2std_valid).all() and np.isfinite(lower_band_2std_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_2std_valid, upper_band_2std_valid, color='gold', alpha=0.2, label='+/- 2 Std Dev Band') # Use a lighter color/alpha
                 else:
                      st.warning("+/- 2 Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")

                 # Plot +/- 1 Std Dev Band SECOND
                 if np.isfinite(upper_band_valid).all() and np.isfinite(lower_band_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_valid, upper_band_valid, color='darkorange', alpha=0.3, label='+/- 1 Std Dev Band') # Use darker color/alpha
                 else:
                      st.warning("+/- 1 Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")


                 # Plot Median line LAST (on top)
                 ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations_ran} runs)', color='red', linestyle='-', linewidth=2)


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
                                   color='goldenrod', fontsize=9, ha='left', va='bottom') # Use color matching +/- 2 band fill

                      if np.isfinite(lower_band_2std[last_valid_step_index_in_simulation]):
                            ax.text(final_date_text, lower_band_2std[last_valid_step_index_in_simulation], f" ${lower_band_2std[last_valid_step_index_in_simulation]:.2f}",
                                   color='goldenrod', fontsize=9, ha='left', va='top') # Use color matching +/- 2 band fill


                 else:
                     st.warning("Median simulated data contains no finite points, skipping end labels.")

                 ax.legend()
              else:
                   st.warning("No finite aggregate simulation data points available to plot the median line/bands.")


         else:
             st.warning("Could not plot simulation aggregates. Data length issues or all aggregate values are non-finite.")
             # Still plot historical data even if simulation plot fails
             # The historical plot is already done above
    # else: # simulation_results is None, so no simulation has been run yet or it failed
        # st.info("Click 'Run Simulation' to see forecasts.") # Optional message


    # Set plot title to reflect displayed historical range vs analysis range
    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)}/{historical_data_close_analyzed_len} days analyzed) & Simulation (EWMA $\lambda$={ewma_lambda}, SMA Bias Period={sma_period}, Multiplier={bias_multiplier})' # Updated title
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d')) # Optional: format dates
    #fig.autofmt_xdate() # Optional: tilt dates for readability

    # Return the figure object
    return fig


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


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis Calculation, Slider Definition) runs on every rerun.


# --- Button to Run Simulation ---
# This block runs ONLY when the button is clicked
if st.button("Run Simulation"):
    # Clear previous simulation results from session state
    st.session_state.simulation_results = None

    # --- Calculate Simulation Aggregates (Heavy Computation) ---
    # This happens once per button click
    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days with EWMA $\lambda$={ewma_lambda}, SMA Period={sma_period}, Bias Multiplier={bias_multiplier}..."):

        # Get start price from the data used for analysis
        # Ensure historical_data_close_analyzed is not empty before trying to get the last price
        # The data needed is historical_data_close_analyzed (tail of full_historical_data)
        if historical_data_close_analyzed.empty:
            st.error("Cannot run simulation: Historical data for analysis is empty.")
            st.session_state.simulation_results = None
            st.stop()

        try:
            # Use .iloc[-1].item() to explicitly get the scalar value and avoid FutureWarning
            start_price = float(historical_data_close_analyzed.iloc[-1].item())
        except (TypeError, ValueError) as e:
             st.error(f"Unexpected value or type for last historical price: {e}")
             start_price = np.nan # Set to NaN if conversion fails

        if not np.isfinite(start_price) or start_price <= 0:
             st.error(f"Last historical price ({start_price}) is not a finite positive number. Cannot start simulation.")
             # Set session state to None to indicate simulation failed/skipped
             st.session_state.simulation_results = None
             st.stop()

        # The calculated `mean_float_for_sim` (biased mean) and `volatility_float_for_sim` (EWMA volatility)
        # are available from the historical statistics calculation block above.

        loc_sim = mean_float_for_sim # Use the calculated and checked biased mean
        scale_sim = volatility_float_for_sim # Use the calculated and checked EWMA volatility


        # Prepare dates for simulation results plotting (based on analysis dates)
        historical_dates_analysis = historical_data_close_analyzed.index
        # Use .max() as it's a DatetimeIndex, result is pandas Timestamp
        last_historical_date_analysis = historical_dates_analysis.max()

        simulated_dates_pd = pd.DatetimeIndex([])
        sim_path_length = 0

        try:
            # Generate future dates, handling potential errors
            # Increased periods by 1 to include the day *after* the last historical day as the first simulation day
            simulated_dates_pd = pd.date_range(start=last_historical_date_analysis, periods=simulation_days + 1, freq='B')[1:] # [1:] excludes start date
            if len(simulated_dates_pd) != simulation_days:
                 st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date_analysis.strftime('%Y-%m-%d')}. Generated {len(simulated_dates_pd)}. Simulation path length will be adjusted.")
            sim_path_length = len(simulated_dates_pd) + 1 # Add 1 for the starting price point
        except Exception as date_range_error:
             st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot run simulation.")
             st.session_state.simulation_results = None
             st.stop()

        # Prepare the full date axis for plotting (Historical End + Simulated Dates)
        plot_sim_dates_pd = pd.DatetimeIndex([])
        if len(simulated_dates_pd) > 0 and sim_path_length > 0:
            # Combine the last historical date with the simulated future dates
            last_historical_date_index = pd.DatetimeIndex([last_historical_date_analysis])
            plot_sim_dates_pd = last_historical_date_index.append(simulated_dates_pd)

            # Check length consistency
            if len(plot_sim_dates_pd) != sim_path_length:
                 st.error(f"Mismatch between calculated path length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates_pd)}). Cannot plot.")
                 st.session_state.simulation_results = None
                 st.stop()
        else:
             st.error("Skipping simulation as future dates could not be generated or have zero length.")
             st.session_state.simulation_results = None
             st.stop()


        # --- Run Multiple Simulations (Inner loop) ---
        # Use the calculated (potentially biased) mean (loc_sim) and EWMA volatility (scale_sim)
        # Generate all random log returns for all simulations at once for efficiency
        all_simulated_log_returns = np.random.normal(
             loc=loc_sim,    # Biased Mean drift
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
        upper_band = mean_prices + std_dev_prices # +1 Std Dev
        lower_band = mean_prices - std_dev_prices # -1 Std Dev
        upper_band_2std = mean_prices + 2 * std_dev_prices # New: +2 Std Dev
        lower_band_2std = mean_prices - 2 * std_dev_prices # New: -2 Std Dev


        # Filter final prices to exclude non-finite values (robustness)
        final_prices = [price for price in final_prices_list_raw if np.isfinite(price)]


        # --- Calculate Forecast Metrics (based on +/- 1 std dev) ---
        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        # Ensure last historical price is available and finite (already checked start_price)
        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1].item()) # Use .item()


        # Ensure final aggregate band values are finite for percentage calculation
        # Risk/Reward and +/- 1 Std Dev movement are based on +/- 1 band endpoints
        final_upper_price_1std = upper_band[-1] if len(upper_band) > 0 and np.isfinite(upper_band[-1]) else np.nan
        final_lower_price_1std = lower_band[-1] if len(lower_band) > 0 and np.isfinite(lower_band[-1]) else np.nan

        # Percentage Delta to +1 Std Dev (using the +/- 1 band)
        if np.isfinite(final_upper_price_1std) and last_historical_price_scalar > 0:
             delta_upper_pct = ((final_upper_price_1std - last_historical_price_scalar) / last_historical_price_scalar) * 100

        # Percentage Delta to -1 Std Dev (using the +/- 1 band)
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
    # This makes the results available for the plotting and display functions on the next rerun
    # Store all necessary variables, including the original data used for analysis summary
    st.session_state.simulation_results = {
        'historical_data_close_analyzed': historical_data_close_analyzed, # Store this too for the final table
        'original_mean_float': original_mean_float, # New: Store original mean
        'biased_mean_float_used': biased_mean_float,     # New: Store the biased mean used
        'sma_period_used': sma_period,              # New: Store SMA period used
        'bias_multiplier_used': bias_multiplier,    # New: Store bias multiplier used
        'volatility_float': volatility_float, # Store historical stats (EWMA)
        'ewma_lambda_used': ewma_lambda, # Store lambda used
        'plot_sim_dates': plot_sim_dates_pd, # Store pandas DatetimeIndex for plotting/display
        'median_prices': median_prices,
        'mean_prices': mean_prices, # Store aggregate mean for the table
        'std_dev_prices': std_dev_prices, # Store aggregate std dev for the table
        'upper_band': upper_band, # Store +1 std dev band
        'lower_band': lower_band, # Store -1 std dev band
        'upper_band_2std': upper_band_2std, # New: Store +2 std dev band
        'lower_band_2std': lower_band_2std, # New: Store -2 std dev band
        'final_prices': final_prices, # List of finite actual final prices
        'simulated_dates': simulated_dates_pd, # Dates for the simulation period (without start)
        'delta_upper_pct': delta_upper_pct, # Store sidebar values (+1 std dev based)
        'delta_lower_pct': delta_lower_pct, # Store sidebar values (-1 std dev based)
        'risk_reward_ratio': risk_reward_ratio, # Store sidebar values (+1/-1 std dev based)
        'num_simulations_ran': num_simulations, # Store number of sims run for display
    }
    #st.success("Simulation completed and results stored.")
    # Note: Streamlit will automatically rerun the script after this block finishes.


# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch, Analysis Calculation, Slider Definition) runs on every rerun.


# --- Display Risk/Reward and Key Forecasts (outside button block) ---
# This runs on every rerun if simulation results are in session state
# Moved from sidebar
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results

    # Use .get() with default values in case keys are missing due to errors
    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    delta_lower_pct = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio = results.get('risk_reward_ratio', np.nan)
    num_simulations_ran = results.get('num_simulations_ran', 'N/A')


    st.subheader("Simulation Forecast Insights")
    st.write(f"*(Based on {num_simulations_ran} runs)*")
    col1, col2, col3 = st.columns(3) # Use columns for better layout

    with col1:
        # Use format_percentage helper function for display
        st.metric(label="Expected movement to +1 Std Dev End", value=format_percentage(delta_upper_pct, '.2f', 'N/A'))
    with col2:
        st.metric(label="Expected movement to -1 Std Dev End", value=format_percentage(delta_lower_pct, '.2f', 'N/A'))
    with col3:
        # Format risk/reward ratio
        risk_reward_str = "N/A"
        if np.isfinite(risk_reward_ratio):
             if risk_reward_ratio == np.inf:
                  risk_reward_str = "Infinite"
             else:
                  risk_reward_str = f"{risk_reward_ratio:.2f} : 1"
        elif np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
              # If ratio is NaN but deltas are finite, it implies delta_lower_pct was >= 0
             risk_reward_str = "Undetermined / Favorable Downside" # e.g. +5% gain, -0.1% loss

        st.metric(label="Risk/Reward Ratio (+1 Gain : -1 Loss)", value=risk_reward_str)


# --- Display Plot (outside button block) ---
# This runs on every rerun (button click or slider move)
# It gets data from full_historical_data (cached) and reads simulation results from session state.
st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")
if full_historical_data is not None and not full_historical_data.empty:
    # Pass required analysis parameters to the plotting function for title
    fig = plot_simulation(
        full_historical_data,
        historical_days_to_display,
        st.session_state.simulation_results,
        ticker, # Pass ticker
        len(historical_data_close_analyzed), # Pass analyzed data count for title
        ewma_lambda, # Pass lambda for title
        sma_period, # Pass SMA period for title
        bias_multiplier # Pass bias multiplier for title
    )
    st.pyplot(fig)
    plt.close(fig) # Always close the figure
else:
    # Display a message if full historical data could not be fetched initially
    if 'full_historical_data' in locals() and (full_historical_data is None or full_historical_data.empty):
         st.error("Cannot display plot because historical data fetching failed.")
    else:
         # This case should ideally be caught by the initial check/st.stop(), but defensive
         st.warning("Waiting for historical data...")


# --- Display Results Table (outside button block, at the bottom) ---
# This runs on every rerun. It displays the table IF simulation results are in session state.
if st.session_state.simulation_results is not None:
    st.subheader("Simulation and Analysis Summary")

    results = st.session_state.simulation_results

    # Get historical stats from stored results
    historical_data_analyzed = results.get('historical_data_close_analyzed')
    hist_analysis_days = len(historical_data_analyzed) if isinstance(historical_data_analyzed, pd.Series) else 0 # Safe length check
    original_mean_log_return = results.get('original_mean_float', np.nan) # New
    biased_mean_log_return_used = results.get('biased_mean_float_used', np.nan) # New
    sma_period_used = results.get('sma_period_used', 'N/A') # New
    bias_multiplier_used = results.get('bias_multiplier_used', 'N/A') # New
    hist_volatility_log = results.get('volatility_float', np.nan) # This is the EWMA volatility
    ewma_lambda_used = results.get('ewma_lambda_used', 'N/A')

    # --- SAFELY GET LAST HISTORICAL PRICE AND DATE ---
    last_historical_price_scalar = np.nan
    last_historical_date_analysis = "N/A" # Keep initial N/A

    # Ensure historical_data_analyzed is a non-empty Series before accessing iloc
    if isinstance(historical_data_analyzed, pd.Series) and not historical_data_analyzed.empty:
        try:
            # Get the last value safely using .item()
            raw_last_value = historical_data_analyzed.iloc[-1]
            # Check if it's finite before converting
            if np.isfinite(raw_last_value):
                last_historical_price_scalar = float(raw_last_value.item()) # Use .item()
                # Get the last date safely using .item() if needed, or directly if it's a Timestamp
                last_historical_date_analysis = historical_data_analyzed.index[-1] # This returns a pandas Timestamp

            else:
                 # Handle cases where the last price is NaN or Inf
                 st.warning("Last historical price is non-finite (NaN/Inf), cannot display in table.")
        except Exception as e:
            st.error(f"Error processing last historical price or date for table: {e}")
            # Keep them as np.nan / "N/A"


    # Get simulation results from stored results
    num_sims_ran = results.get('num_simulations_ran', 'N/A')
    sim_dates_pd = results.get('simulated_dates', pd.DatetimeIndex([])) # Get the pandas DatetimeIndex
    sim_days = len(sim_dates_pd) # Number of steps into the future (length of simulated_dates)
    # Safely get the last simulation date using max() on the DatetimeIndex
    sim_end_date = sim_dates_pd.max() if len(sim_dates_pd) > 0 else "N/A"


    # Get aggregate results arrays safely
    median_prices_array = results.get('median_prices', np.array([]))
    mean_prices_array = results.get('mean_prices', np.array([]))
    std_dev_prices_array = results.get('std_dev_prices', np.array([]))
    upper_band_array = results.get('upper_band', np.array([])) # +1 Std Dev
    lower_band_array = results.get('lower_band', np.array([])) # -1 Std Dev
    upper_band_2std_array = results.get('upper_band_2std', np.array([])) # New: +2 Std Dev
    lower_band_2std_array = results.get('lower_band_2std', np.array([])) # New: -2 Std Dev
    final_prices_list = results.get('final_prices', []) # List of finite actual final prices

    # Safely get final values from aggregate arrays
    median_end_price = median_prices_array[-1] if len(median_prices_array) > 0 and np.isfinite(median_prices_array[-1]) else np.nan
    mean_end_price = mean_prices_array[-1] if len(mean_prices_array) > 0 and np.isfinite(mean_prices_array[-1]) else np.nan
    std_dev_end = std_dev_prices_array[-1] if len(std_dev_prices_array) > 0 and np.isfinite(std_dev_prices_array[-1]) else np.nan
    upper_band_end_price_1std = upper_band_array[-1] if len(upper_band_array) > 0 and np.isfinite(upper_band_array[-1]) else np.nan
    lower_band_end_price_1std = lower_band_array[-1] if len(lower_band_array) > 0 and np.isfinite(lower_band_array[-1]) else np.nan
    upper_band_end_price_2std = upper_band_2std_array[-1] if len(upper_band_2std_array) > 0 and np.isfinite(upper_band_2std_array[-1]) else np.nan # New: +2 Std Dev end price
    lower_band_end_price_2std = lower_band_2std_array[-1] if len(lower_band_2std_array) > 0 and np.isfinite(lower_band_2std_array[-1]) else np.nan # New: -2 Std Dev end price


    # Safely get min/max from the list of *actual* final prices
    actual_min_end_price = np.min(final_prices_list) if final_prices_list else np.nan
    actual_max_end_price = np.max(final_prices_list) if final_prices_list else np.nan

    # Get delta percentages and risk/reward from stored values (calculated based on +/- 1 std dev)
    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    delta_lower_pct = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio_val = results.get('risk_reward_ratio', np.nan)


    # Prepare data for the table
    # Use helper function or logic to format values safely, handling np.nan
    # These helper functions are now defined BEFORE they are called


    table_data = {
        'Metric': [
            'Historical Analysis Period (days)',
            'Historical Mean Daily Log Return (Original)', # Updated label
            f'EWMA Daily Log Volatility ($\lambda$={ewma_lambda_used})',
            'Last Historical Price ($)', # Added this for clarity
            'Last Historical Date',      # Added this for clarity
            '--- Bias Settings Used ---', # New separator
            f'SMA Period for Bias ({sma_period_used} days)', # New row
            f'Bias Multiplier ({bias_multiplier_used})', # New row
            f'Biased Mean Daily Log Return (Used in Sim)', # New row
            '--- Simulation Results ---',
            'Number of Simulations',
            # Reference the last historical date safely
            f'Simulation Period (days from {(last_historical_date_analysis.strftime("%Y-%m-%d") if isinstance(last_historical_date_analysis, (datetime, pd.Timestamp)) else str(last_historical_date_analysis))})',
            f'Simulation End Date',
            'Simulated Median Ending Price ($)',
            'Simulated Mean Ending Price ($)',
            'Simulated Std Dev Ending Price ($)',
            'Simulated +1 Std Dev Ending Price ($)', # +/- 1 Std Dev
            'Simulated -1 Std Dev Ending Price ($)',
            'Simulated +2 Std Dev Ending Price ($)', # New: +/- 2 Std Dev
            'Simulated -2 Std Dev Ending Price ($)',
            'Actual Min Simulated Ending Price ($)',
            'Actual Max Simulated Ending Price ($)',
            'Expected movement to +1 Std Dev End (%)', # Keep in table for summary
            'Expected movement to -1 Std Dev End (%)', # Keep in table for summary
            'Risk/Reward Ratio (+1 Gain : -1 Loss)', # Keep in table for summary
        ],
        'Value': [
            hist_analysis_days,
            format_value(original_mean_log_return, ".6f"), # Use original mean
            format_value(hist_volatility_log, ".6f"),
            format_value(last_historical_price_scalar, ".2f"), # Formatted safely
            last_historical_date_analysis.strftime('%Y-%m-%d') if isinstance(last_historical_date_analysis, (datetime, pd.Timestamp)) else str(last_historical_date_analysis), # Date format check
            '', # Separator
            sma_period_used, # Display SMA period
            bias_multiplier_used, # Display bias multiplier
            format_value(biased_mean_log_return_used, ".6f"), # Display biased mean
            '', # Separator
            num_sims_ran,
            sim_days,
            sim_end_date.strftime('%Y-%m-%d') if isinstance(sim_end_date, (datetime, pd.Timestamp)) else str(sim_end_date), # Date format check
            format_value(median_end_price, ".2f"),
            format_value(mean_end_price, ".2f"),
            format_value(std_dev_end, ".2f"),
            format_value(upper_band_end_price_1std, ".2f"), # Use _1std variable
            format_value(lower_band_end_price_1std, ".2f"), # Use _1std variable
            format_value(upper_band_end_price_2std, ".2f"), # New: Use _2std variable
            format_value(lower_band_end_price_2std, ".2f"), # New: Use _2std variable
            format_value(actual_min_end_price, ".2f"),
            format_value(actual_max_end_price, ".2f"),
            format_percentage(delta_upper_pct, ".2f"), # Use helper
            format_percentage(delta_lower_pct, ".2f"), # Use helper
            f"{format_value(risk_reward_ratio_val, '.2f')} : 1" if np.isfinite(risk_reward_ratio_val) and risk_reward_ratio_val != np.inf else ("Infinite" if risk_reward_ratio_val == np.inf else "N/A"),
        ]
    }

    # Create DataFrame and display
    results_df = pd.DataFrame(table_data)
    st.dataframe(results_df, hide_index=True, use_container_width=True)
