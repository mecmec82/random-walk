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
    2.  **Analysis Period:** You select a number of historical trading days (`Historical Trading Days for Analysis`). The app uses the closing prices from *this specific period* ending on the simulation start date to calculate:
        *   The average daily log return (drift).
        *   The **daily volatility** using an **Exponentially Weighted Moving Average (EWMA)** method. EWMA gives more importance to recent price movements based on the `EWMA Decay Factor (Lambda)`. A lambda closer to 1 means recent data are weighted much higher. This volatility is crucial as it determines the *size* of the random steps in the simulation.
    3.  **Mean Bias Feature (Optional):**
        *   You can introduce a bias to the calculated historical mean daily return based on whether the last price is above or below a Simple Moving Average (SMA).
        *   If the last price is above the SMA (over the `SMA Period for Bias`), the mean drift used in the simulation is slightly increased. If it's below the SMA, the mean drift is slightly decreased.
        *   The magnitude of the bias is controlled by the `Bias Multiplier` multiplied by the absolute value of the original historical mean return. A multiplier of 0 means no bias is applied.
        *   This requires fetching historical data covering the `Analysis Period` plus the `SMA Period` before the simulation start date.
    4.  **Random Walk Simulation:** For each simulation step into the future (`Future Simulation Days`), the model calculates the next price based on the previous day's price multiplied by a random factor. This random factor is determined by:
        *   The calculated mean drift (potentially biased).
        *   A random value drawn from a **normal (Gaussian) distribution**.
        *   The size of this random value is scaled by the calculated EWMA daily volatility.
    5.  **Monte Carlo Approach:** The simulation is repeated many times (`Number of Simulations to Run`). Each simulation run generates a different possible price path because of the random component.
    6.  **Aggregation and Results:**
        *   Instead of showing hundreds of individual paths (which would be messy), the app calculates the **median** price across all simulations at each future time step. This gives a sense of the most 'typical' outcome.
        *   It also calculates **standard deviation bands** (+/- 1 and +/- 2 standard deviations) around the mean price at each step. These bands indicate the typical spread of outcomes. Roughly 68% of simulated paths are expected to stay within the +/- 1 Std Dev band, and 95% within the +/- 2 Std Dev band, *if the underlying assumptions held perfectly*.
        *   Key metrics like the expected price movement to the edge of the +/- 1 Std Dev band and a Risk/Reward ratio based on the +/- 1 Std Dev endpoints are calculated from the simulation's final step aggregates.
    7.  **Plotting and Summary:** The historical data (you can adjust how many historical days are *displayed* on the plot using a separate slider, without changing the analysis period) is plotted alongside the simulated median path and the standard deviation bands. A table provides a summary of the historical analysis parameters and the key simulation results.

    **Backtesting Feature (Optional):**

    *   The backtesting feature allows you to see how often the model's +/- 1 Standard Deviation projection band for the *ending price* would have contained the *actual* price `Future Simulation Days` later, over a chosen historical period (`Backtest Period (Days)`).
    *   It runs a simulation for *each day* within the backtest period.
    *   A higher "Pass Ratio" suggests the model's volatility and drift estimates are historically good indicators of the typical price range, but it does *not* guarantee future accuracy.

    **Important Considerations:**

    *   **This is a simplified model:** It assumes future price movements are *random* and follow a normal distribution with constant drift and *predictable* volatility (based on EWMA), plus a simple SMA-based directional bias on the mean.
    *   **It does NOT predict the future:** It shows a *range of possible outcomes* based on historical patterns and random chance. Real markets are influenced by news, events, changing fundamentals, and human behavior that are not captured by this simple random walk.
    *   **Volatility is not constant:** While EWMA is better than a simple average, volatility still changes in ways not fully captured (e.g., GARCH models attempt this).
    *   **Returns may not be normally distributed:** Extreme events happen more often than the normal distribution predicts ("fat tails").
    *   Use these results for exploring potential scenarios and risk, not as definitive predictions.
    """)

# --- Initialize Session State for Simulation Results & Backtest Results ---
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = None
if 'backtest_results' not in st.session_state:
     st.session_state.backtest_results = None


# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

ticker = st.sidebar.text_input("Ticker Symbol (e.g., SPY, BTC-USD)", 'BTC-USD').upper()

# Number input for historical days used for ANALYSIS
# This input triggers a rerun and cached data fetch/analysis
historical_days_analysis = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=100, value=900, step=100, help="Number of recent historical days used to calculate mean and volatility.")

simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1, help="Number of days to simulate into the future.")
num_simulations = st.sidebar.number_input("Number of Simulations to Run", min_value=1, value=10000, step=10, help="Number of Monte Carlo paths for the main simulation.")

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
bias_multiplier = st.sidebar.number_input("Bias Multiplier", min_value=0.0, value=0.5, step=0.05, format="%.2f", help="Multiplier for biasing the mean drift. If price is above SMA, mean increases by this factor * abs(mean). If below, decreases. 0 means no bias.")
st.sidebar.markdown(f"*(Bias amount = `{bias_multiplier} * |Historical Mean|`)*")

# --- Backtesting Settings ---
st.sidebar.header("Backtesting Settings")
backtest_days_count = st.sidebar.number_input("Backtest Period (Days)", min_value=30, value=252, step=30, help="Number of past trading days to use as simulation start points for backtesting.")
num_simulations_backtest = st.sidebar.number_input("Simulations per Backtest Point", min_value=50, value=500, step=50, help="Number of Monte Carlo paths per day during backtesting (lower value speeds up backtest).")


st.sidebar.write("Data fetched using yfinance (Yahoo Finance).")
st.sidebar.write("Volatility estimated using Exponentially Weighted Moving Average (EWMA).")
st.sidebar.write("Note: Yahoo Finance data may have occasional inaccuracies or downtime.")


# --- Helper function to fetch historical data using yfinance ---
# Cached function: runs only when ticker, historical_days_analysis, sma_period, or backtest_days_count changes
# Need enough data for the full backtest period PLUS the analysis window (analysis_days + sma_period)
@st.cache_data(ttl=3600) # Cache data fetching results for 1 hour (3600 seconds)
def fetch_full_historical_data(ticker, analysis_days, sma_period, backtest_days):
    """
    Fetches enough historical data to cover the backtest period + analysis window.
    Returns a pandas Series of closing prices with datetime index.
    """
    # We need data going back `backtest_days` + `analysis_days` + `sma_period` from today.
    # Use a generous estimate (3x) for calendar days to ensure we get enough trading days
    total_days_needed = backtest_days + analysis_days + sma_period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(total_days_needed * 3.0)) # Increased buffer


    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series()

        historical_data_close = data['Close']

        if not isinstance(historical_data_close.index, pd.DatetimeIndex):
             try:
                  historical_data_close.index = pd.to_datetime(historical_data_close.index)
             except Exception as e:
                  st.error(f"Error converting historical data index to DatetimeIndex: {e}")
                  return pd.Series()

        historical_data_close = historical_data_close.sort_index()

        original_len = len(historical_data_close)
        historical_data_close = historical_data_close[historical_data_close > 0]
        if len(historical_data_close) < original_len:
             st.warning(f"Filtered out {original_len - len(historical_data_close)} rows with non-positive prices.")
        if historical_data_close.empty:
             st.error("No historical data with positive prices found.")
             return pd.Series()

        # Return the full fetched & filtered Series
        return historical_data_close

    except Exception as e:
        st.error(f"Error fetching full historical data for {ticker} using yfinance: {e}")
        st.error("Please check the ticker symbol and your internet connection.")
        return pd.Series()

# --- Fetch Full Historical Data (Cached) ---
full_historical_data = fetch_full_historical_data(
    ticker=ticker,
    analysis_days=historical_days_analysis,
    sma_period=sma_period,
    backtest_days=backtest_days_count # Needed for fetching sufficient history
)

# --- Core Calculation Function (Reusable) ---
# This function calculates stats for a specific window of historical data
def calculate_historical_stats(historical_data_window, analysis_days, sma_period, bias_multiplier, ewma_lambda):
    """
    Calculates original mean, biased mean, and EWMA volatility for a given
    historical data window ending on the simulation start date.
    """
    # Ensure the window has enough data for analysis + SMA
    min_required_window = analysis_days + sma_period
    if len(historical_data_window) < min_required_window:
         # Not enough history in this specific window for calculations
         return None, None, None, None, None, f"Insufficient data in window ({len(historical_data_window)} days), need {min_required_window} for analysis+SMA"


    # 1. Isolate data for the analysis period (tail of the window)
    historical_data_close_analyzed = historical_data_window.tail(analysis_days)
    if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
         return None, None, None, None, None, "Insufficient data in analysis period"

    # Calculate log returns for the analysis period
    log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()
    if len(log_returns) < 1:
        return None, None, None, None, None, "Insufficient returns for stats calculation"

    # 2. Calculate Original Mean Daily Log Return
    original_mean_float = float(log_returns.mean().item())

    # 3. Calculate EWMA Daily Log Volatility
    ewma_alpha = 1 - ewma_lambda
    daily_log_volatility = np.nan # Initialize

    try:
        ewma_variance_series = log_returns.astype(float).pow(2).ewm(alpha=ewma_alpha, adjust=False, min_periods=0).mean()
        ewma_daily_log_volatility_scalar = np.sqrt(ewma_variance_series.iloc[-1]).item()
        daily_log_volatility = float(ewma_daily_log_volatility_scalar)
    except Exception as e:
        return None, None, None, None, None, f"Error calculating EWMA volatility: {e}"

    # 4. Calculate SMA and Determine Bias
    biased_mean_float = original_mean_float # Start with the original mean

    # Calculate the SMA over the full historical_data_window
    sma_series = historical_data_window.rolling(window=sma_period).mean().dropna()

    # Ensure SMA series has a value corresponding to the end of the analysis period
    last_date_analyzed = historical_data_close_analyzed.index.max() # Last date of the analysis period

    sma_bias_applied = False
    sma_calc_message = ""

    if not sma_series.empty and last_date_analyzed in sma_series.index:
         try:
              # Get the last price from the data used for ANALYSIS
              last_price_for_sma_comp = historical_data_close_analyzed.iloc[-1].item()
              # Get the SMA value corresponding to the last date in the analysis period
              last_sma_value = sma_series.loc[last_date_analyzed].item()

              bias_amount = bias_multiplier * abs(original_mean_float)

              if last_price_for_sma_comp > last_sma_value:
                  biased_mean_float = original_mean_float + bias_amount
                  sma_calc_message = f"Last price ABOVE SMA ({last_price_for_sma_comp:.2f} > {last_sma_value:.2f}). UPWARD bias applied ({bias_amount:.6f})."
                  sma_bias_applied = True
              elif last_price_for_sma_comp < last_sma_value:
                  biased_mean_float = original_mean_float - bias_amount
                  sma_calc_message = f"Last price BELOW SMA ({last_price_for_sma_comp:.2f} < {last_sma_value:.2f}). DOWNWARD bias applied (-{bias_amount:.6f})."
                  sma_bias_applied = True
              else:
                  biased_mean_float = original_mean_float
                  sma_calc_message = f"Last price EQUAL to SMA ({last_price_for_sma_comp:.2f} == {last_sma_value:.2f}). No bias applied."
                  sma_bias_applied = True # Calculation was performed

         except Exception as e:
              sma_calc_message = f"Error during SMA comparison or bias calculation: {e}. Using original mean."
              biased_mean_float = original_mean_float # Ensure it's the original mean if bias fails
    else:
        sma_calc_message = f"Not enough data for {sma_period}-day SMA calculation ending at {last_date_analyzed.strftime('%Y-%m-%d')}. Using original mean."
        biased_mean_float = original_mean_float # Ensure it's the original mean if bias fails


    # 5. Final Validation and Return
    if np.isfinite(biased_mean_float) and np.isfinite(daily_log_volatility) and daily_log_volatility > 0:
         return original_mean_float, biased_mean_float, daily_log_volatility, sma_bias_applied, sma_calc_message, None # Return calculated values and no error message
    else:
         return None, None, None, None, None, "Final calculated mean/volatility is non-finite or volatility is not positive." # Return error message


# --- Core Simulation Runner Function (Reusable) ---
# This function runs the Monte Carlo simulation itself
def run_monte_carlo_simulation(start_price, biased_mean, volatility, simulation_days, num_simulations_per_step):
    """
    Runs Monte Carlo simulations and calculates aggregate paths.
    Returns median, mean, std dev paths, and final endpoints.
    """
    if not np.isfinite(start_price) or start_price <= 0:
        return None, None, None, None, None, None, None, "Invalid start price for simulation."
    if not np.isfinite(biased_mean) or not np.isfinite(volatility) or volatility <= 0:
         return None, None, None, None, None, None, None, "Invalid mean/volatility for simulation."
    if simulation_days <= 0 or num_simulations_per_step <= 0:
         return None, None, None, None, None, None, None, "Invalid simulation parameters (days/count)."


    sim_path_length = simulation_days + 1 # Add 1 for the starting point

    # Generate all random log returns for all simulations at once for efficiency
    all_simulated_log_returns = np.random.normal(
         loc=biased_mean,    # Biased Mean drift
         scale=volatility, # EWMA Volatility
         size=(num_simulations_per_step, simulation_days) # Shape is (num_simulations, num_steps)
    )

    # Calculate simulated price paths efficiently using cumulative product
    start_prices_array = np.full((num_simulations_per_step, 1), start_price)
    price_change_factors = np.exp(all_simulated_log_returns)
    cumulative_price_multipliers = np.cumprod(np.concatenate((np.ones((num_simulations_per_step, 1)), price_change_factors), axis=1), axis=1)
    all_simulated_paths_np = start_prices_array * cumulative_price_multipliers

    # --- Calculate Median, Mean, Standard Deviation Across Simulations ---
    # Transpose the array so each row is a timestep, and columns are simulations
    prices_at_each_step = all_simulated_paths_np.T # Shape (sim_path_length, num_simulations)

    # Use nan functions in case any paths resulted in NaN/Inf (shouldn't happen with finite input)
    median_prices = np.nanmedian(prices_at_each_step, axis=1)
    mean_prices = np.nanmean(prices_at_each_step, axis=1)
    std_dev_prices = np.nanstd(prices_at_each_step, axis=1)

    # Calculate upper/lower bands based on mean +/- std dev
    upper_band_1std = mean_prices + std_dev_prices # +1 Std Dev
    lower_band_1std = mean_prices - std_dev_prices # -1 Std Dev
    upper_band_2std = mean_prices + 2 * std_dev_prices # +2 Std Dev
    lower_band_2std = mean_prices - 2 * std_dev_prices # -2 Std Dev

    # Return aggregate paths and final endpoints
    return median_prices, mean_prices, std_dev_prices, upper_band_1std, lower_band_1std, upper_band_2std, lower_band_2std, None # Return None for error message if successful


# --- Add Slider for Displayed Historical Days ---
# This slider is defined *outside* the button block so it appears immediately.
# Its max_value is based on the actual number of days successfully fetched and available for analysis.
# historical_data_close_analyzed needs to be calculated before the slider runs.
# We need historical_data_for_sma_and_analysis first.
# Then we take the tail of that for the analysis period.
historical_data_close_analyzed_initial = full_historical_data.tail(historical_days_analysis) # Data for analysis period based on current sidebar input

max_display_days = len(historical_data_close_analyzed_initial) # Use the length of the *analysis period* data
# Set default value: requested days, capped by available, min 100 unless less data is available
default_display_days = min(historical_days_analysis, max_display_days)
default_display_days = max(100, default_display_days) if max_display_days >= 100 else max_display_days
default_display_days = max(1, default_display_days) # Ensure at least 1 day can be displayed if data is minimal


historical_days_to_display = st.sidebar.slider(
    "Historical Days to Display on Plot",
    min_value=min(1, max_display_days) if max_display_days > 0 else 0, # Ensure min_value is valid
    max_value=max_display_days,
    value=default_display_days,
    step=10,
    help="Slide to change the number of historical trading days shown on the chart. Does not affect analysis period or calculated volatility/bias."
)


# --- Helper functions for formatting values safely ---
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
def plot_simulation(full_historical_data, historical_days_to_display, simulation_results, ticker, historical_analysis_len, ewma_lambda, sma_period, bias_multiplier):
    fig, ax = plt.subplots(figsize=(14, 7))

    # --- Select data for plotting using the slider value and convert to numpy arrays ---
    historical_data_to_plot = full_historical_data.tail(historical_days_to_display)
    historical_dates_to_plot_np = historical_data_to_plot.index.values if not historical_data_to_plot.empty else np.array([])
    historical_prices_to_plot_np = historical_data_to_plot.values if not historical_data_to_plot.empty else np.array([])

    # Plot Historical Data
    if len(historical_prices_to_plot_np) > 0:
        ax.plot(historical_dates_to_plot_np, historical_prices_to_plot_np, label=f'Historical {ticker} Price ({len(historical_data_to_plot)} days displayed)', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    # Plot Aggregated Simulated Data (Median and Band)
    if simulation_results is not None:
         plot_sim_dates_pd = simulation_results.get('plot_sim_dates')
         median_prices = simulation_results.get('median_prices')
         upper_band = simulation_results.get('upper_band') # +1 std dev
         lower_band = simulation_results.get('lower_band') # -1 std dev
         upper_band_2std = simulation_results.get('upper_band_2std') # +2 std dev
         lower_band_2std = simulation_results.get('lower_band_2std') # -2 std dev
         num_simulations_ran = simulation_results.get('num_simulations_ran', 'N/A')

         plot_sim_dates_np = plot_sim_dates_pd.values if plot_sim_dates_pd is not None else np.array([])

         if (len(plot_sim_dates_np) > 0 and
             median_prices is not None and len(median_prices) == len(plot_sim_dates_np) and np.isfinite(median_prices).any() and
             upper_band is not None and len(upper_band) == len(plot_sim_dates_np) and
             lower_band is not None and len(lower_band) == len(plot_sim_dates_np) and
             upper_band_2std is not None and len(upper_band_2std) == len(plot_sim_dates_np) and
             lower_band_2std is not None and len(lower_band_2std) == len(plot_sim_dates_np)):

              valid_plot_indices = np.isfinite(median_prices)
              plot_sim_dates_valid = plot_sim_dates_np[valid_plot_indices]
              median_prices_valid = median_prices[valid_plot_indices]
              upper_band_valid = upper_band[valid_plot_indices]
              lower_band_valid = lower_band[valid_plot_indices]
              upper_band_2std_valid = upper_band_2std[valid_plot_indices]
              lower_band_2std_valid = lower_band_2std[valid_plot_indices]

              if len(plot_sim_dates_valid) > 0:
                 # Plot +/- 2 Std Dev Band
                 if np.isfinite(upper_band_2std_valid).all() and np.isfinite(lower_band_2std_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_2std_valid, upper_band_2std_valid, color='gold', alpha=0.2, label='+/- 2 Std Dev Band')
                 else:
                      st.warning("+/- 2 Std dev band contains non-finite values. Not plotting band.")

                 # Plot +/- 1 Std Dev Band
                 if np.isfinite(upper_band_valid).all() and np.isfinite(lower_band_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_valid, upper_band_valid, color='darkorange', alpha=0.3, label='+/- 1 Std Dev Band')
                 else:
                      st.warning("+/- 1 Std dev band contains non-finite values. Not plotting band.")

                 # Plot Median line
                 ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations_ran} runs)', color='red', linestyle='-', linewidth=2)

                 # --- Add Labels at the end ---
                 last_valid_step_index_in_simulation = len(median_prices) - 1
                 while last_valid_step_index_in_simulation >= 0 and not np.isfinite(median_prices[last_valid_step_index_in_simulation]):
                      last_valid_step_index_in_simulation -= 1

                 if last_valid_step_index_in_simulation >= 0:
                      final_date_text = plot_sim_dates_pd[last_valid_step_index_in_simulation]
                      final_median_price = median_prices[last_valid_step_index_in_simulation]

                      ax.text(final_date_text, final_median_price, f" ${final_median_price:.2f}",
                              color='red', fontsize=10, ha='left', va='center', weight='bold')

                      if np.isfinite(upper_band[last_valid_step_index_in_simulation]):
                           ax.text(final_date_text, upper_band[last_valid_step_index_in_simulation], f" ${upper_band[last_valid_step_index_in_simulation]:.2f}",
                                   color='darkorange', fontsize=9, ha='left', va='bottom')
                      if np.isfinite(lower_band[last_valid_step_index_in_simulation]):
                            ax.text(final_date_text, lower_band[last_valid_step_index_in_simulation], f" ${lower_band[last_valid_step_index_in_simulation]:.2f}",
                                   color='darkorange', fontsize=9, ha='left', va='top')
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

    # Set plot title to reflect displayed historical range vs analysis range and bias settings
    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)} days displayed, {historical_analysis_len} days analyzed) & Simulation (EWMA $\lambda$={ewma_lambda}, SMA Bias Period={sma_period}, Multiplier={bias_multiplier})'
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)

    return fig

# --- Main Execution Flow (runs on every rerun) ---

# The code up to here (Inputs, Fetch Full Data, Check Sufficient Data, Slider Definition) runs on every rerun.

# Calculate initial historical stats for display (based on the latest window)
# This provides immediate feedback on the calculated mean/volatility/bias before clicking 'Run Simulation'
historical_data_for_sma_and_analysis_initial = full_historical_data.tail(historical_days_analysis + sma_period)
initial_original_mean, initial_biased_mean, initial_volatility, initial_sma_applied, initial_sma_message, initial_calc_error = \
    calculate_historical_stats(
        historical_data_for_sma_and_analysis_initial,
        historical_days_analysis,
        sma_period,
        bias_multiplier,
        ewma_lambda
    )

# Display initial stats calculated from the latest data
st.subheader("Latest Historical Analysis")
if initial_calc_error:
     st.error(f"Could not calculate historical statistics from the latest data: {initial_calc_error}")
else:
     st.write(f"Based on the last **{historical_days_analysis}** trading days (plus {sma_period} for SMA lookback):")
     col_stats1, col_stats2 = st.columns(2)
     with col_stats1:
          st.info(f"Original Mean Daily Log Return: `{format_value(initial_original_mean, '.6f', 'N/A')}`")
          st.info(f"EWMA Daily Log Volatility (λ={ewma_lambda}): `{format_value(initial_volatility, '.6f', 'N/A')}`")
     with col_stats2:
          st.info(f"Biased Mean Daily Log Return: `{format_value(initial_biased_mean, '.6f', 'N/A')}`")
          st.info(f"SMA Bias Applied: {'Yes' if initial_sma_applied else 'No'}")
     if initial_sma_message:
          st.caption(initial_sma_message)


# --- Button to Run Main Simulation ---
if st.button("Run Main Simulation"):
    st.session_state.simulation_results = None # Clear previous simulation results
    st.session_state.backtest_results = None # Clear previous backtest results

    # Ensure historical data for analysis is available and sufficient
    historical_data_for_sma_and_analysis_latest = full_historical_data.tail(historical_days_analysis + sma_period)
    min_required_days = historical_days_analysis + sma_period
    if historical_data_for_sma_and_analysis_latest.empty or len(historical_data_for_sma_and_analysis_latest) < min_required_days:
        st.error(f"Insufficient historical data ({len(historical_data_for_sma_and_analysis_latest)} days) for analysis. Need at least {min_required_days} days.")
        st.stop()


    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} days with EWMA λ={ewma_lambda}, SMA Period={sma_period}, Bias Multiplier={bias_multiplier}..."):

        # 1. Calculate Historical Stats for the latest period
        original_mean, biased_mean, volatility, sma_applied, sma_message, calc_error = \
            calculate_historical_stats(
                historical_data_for_sma_and_analysis_latest,
                historical_days_analysis,
                sma_period,
                bias_multiplier,
                ewma_lambda
            )

        if calc_error:
             st.error(f"Error calculating historical statistics for main simulation: {calc_error}")
             st.session_state.simulation_results = None
             st.stop()

        # Get the start price (last price of the analysis period)
        start_price_data = historical_data_for_sma_and_analysis_latest.tail(historical_days_analysis).iloc[-1]
        start_price = float(start_price_data.item()) if np.isfinite(start_price_data) else np.nan

        if not np.isfinite(start_price) or start_price <= 0:
             st.error(f"Invalid start price ({start_price}) for main simulation.")
             st.session_state.simulation_results = None
             st.stop()


        # 2. Run Monte Carlo Simulation
        median_prices, mean_prices, std_dev_prices, upper_band_1std, lower_band_1std, upper_band_2std, lower_band_2std, sim_error = \
             run_monte_carlo_simulation(start_price, biased_mean, volatility, simulation_days, num_simulations)

        if sim_error:
             st.error(f"Error running Monte Carlo simulation: {sim_error}")
             st.session_state.simulation_results = None
             st.stop()

        # 3. Prepare Dates for Plotting
        historical_data_close_analyzed_latest = historical_data_for_sma_and_analysis_latest.tail(historical_days_analysis)
        last_historical_date_analysis = historical_data_close_analyzed_latest.index.max()

        simulated_dates_pd = pd.DatetimeIndex([])
        sim_path_length = simulation_days + 1

        try:
            # Generate future dates, handling potential errors
            simulated_dates_pd = pd.date_range(start=last_historical_date_analysis, periods=simulation_days + 1, freq='B')[1:]
            if len(simulated_dates_pd) != simulation_days:
                 st.warning(f"Could not generate exactly {simulation_days} business days after {last_historical_date_analysis.strftime('%Y-%m-%d')}. Generated {len(simulated_dates_pd)}. Simulation path length might be inconsistent.")
                 sim_path_length = len(simulated_dates_pd) + 1 # Adjust path length

        except Exception as date_range_error:
             st.error(f"Error generating future dates for simulation: {date_range_error}. Cannot plot simulation results.")
             simulated_dates_pd = pd.DatetimeIndex([]) # Ensure empty
             sim_path_length = 0

        plot_sim_dates_pd = pd.DatetimeIndex([])
        if len(simulated_dates_pd) > 0 and sim_path_length > 0:
            last_historical_date_index = pd.DatetimeIndex([last_historical_date_analysis])
            plot_sim_dates_pd = last_historical_date_index.append(simulated_dates_pd)
            if len(plot_sim_dates_pd) != sim_path_length:
                 st.error(f"Mismatch between expected plot length ({sim_path_length}) and generated plot dates length ({len(plot_sim_dates_pd)}). Plotting might fail.")
                 # This is a potential issue if simulated_dates_pd was truncated but sim_path_length wasn't adjusted
                 # Let's ensure they align for plotting
                 sim_path_length = min(sim_path_length, len(plot_sim_dates_pd))
                 # Truncate aggregated paths if necessary to match date length
                 if len(median_prices) > sim_path_length: median_prices = median_prices[:sim_path_length]
                 if len(mean_prices) > sim_path_length: mean_prices = mean_prices[:sim_path_length]
                 if len(std_dev_prices) > sim_path_length: std_dev_prices = std_dev_prices[:sim_path_length]
                 if len(upper_band_1std) > sim_path_length: upper_band_1std = upper_band_1std[:sim_path_length]
                 if len(lower_band_1std) > sim_path_length: lower_band_1std = lower_band_1std[:sim_path_length]
                 if len(upper_band_2std) > sim_path_length: upper_band_2std = upper_band_2std[:sim_path_length]
                 if len(lower_band_2std) > sim_path_length: lower_band_2std = lower_band_2std[:sim_path_length]


        # 4. Calculate Forecast Metrics (+/- 1 std dev based)
        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        if len(upper_band_1std) > 0 and len(lower_band_1std) > 0 and np.isfinite(start_price) and start_price > 0:
             final_upper_price_1std = upper_band_1std[-1] if np.isfinite(upper_band_1std[-1]) else np.nan
             final_lower_price_1std = lower_band_1std[-1] if np.isfinite(lower_band_1std[-1]) else np.nan

             if np.isfinite(final_upper_price_1std):
                 delta_upper_pct = ((final_upper_price_1std - start_price) / start_price) * 100
             if np.isfinite(final_lower_price_1std):
                 delta_lower_pct = ((final_lower_price_1std - start_price) / start_price) * 100

             if np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
                  potential_reward = delta_upper_pct
                  potential_risk_abs = -delta_lower_pct
                  if potential_risk_abs > 1e-9:
                       risk_reward_ratio = potential_reward / potential_risk_abs
                  elif potential_risk_abs >= -1e-9 and potential_reward > 1e-9:
                       risk_reward_ratio = np.inf


        # 5. Store Results in Session State
        st.session_state.simulation_results = {
            'historical_data_close_analyzed': historical_data_close_analyzed_latest,
            'original_mean_float': original_mean,
            'biased_mean_float_used': biased_mean,
            'sma_period_used': sma_period,
            'bias_multiplier_used': bias_multiplier,
            'volatility_float': volatility,
            'ewma_lambda_used': ewma_lambda,
            'plot_sim_dates': plot_sim_dates_pd,
            'median_prices': median_prices,
            'mean_prices': mean_prices,
            'std_dev_prices': std_dev_prices,
            'upper_band': upper_band_1std,
            'lower_band': lower_band_1std,
            'upper_band_2std': upper_band_2std,
            'lower_band_2std': lower_band_2std,
            #'final_prices': final_prices, # Not needed for plot/metrics, just for actual min/max in table
            'simulated_dates': simulated_dates_pd, # Dates for the simulation period (without start)
            'delta_upper_pct': delta_upper_pct,
            'delta_lower_pct': delta_lower_pct,
            'risk_reward_ratio': risk_reward_ratio,
            'num_simulations_ran': num_simulations,
            'start_price': start_price # Store start price for table
        }
        #st.success("Main Simulation completed and results stored.")


# --- Button to Run Backtest ---
if st.button("Run Backtest"):
    st.session_state.simulation_results = None # Clear main simulation results
    st.session_state.backtest_results = None # Clear previous backtest results

    if full_historical_data is None or full_historical_data.empty:
         st.error("Cannot run backtest: No historical data available.")
         st.stop()

    # Determine the range of dates for backtesting
    # Backtest starts from today - simulation_days - backtest_days_count
    # Backtest ends on today - simulation_days
    end_date_backtest = datetime.now() - timedelta(days=1) # End yesterday to ensure future data is available
    start_date_backtest_window = end_date_backtest - timedelta(days=int((backtest_days_count + simulation_days + historical_days_analysis + sma_period) * 1.5)) # Ensure enough historical data before the first backtest date

    # Select data window sufficient for the entire backtest range + lookback
    backtest_data_window = full_historical_data.loc[start_date_backtest_window : end_date_backtest]

    if backtest_data_window.empty:
         st.error(f"No data in backtest window: {start_date_backtest_window.strftime('%Y-%m-%d')} to {end_date_backtest.strftime('%Y-%m-%d')}")
         st.stop()

    # Identify the actual start dates for each simulation in the backtest.
    # These are the dates from `backtest_data_window.index` that allow for:
    # 1. The analysis window (analysis_days + sma_period) *before* the start date.
    # 2. The simulation period (simulation_days) *after* the start date, ending within `full_historical_data`.

    # The minimum required history *before* a backtest start date is analysis_days + sma_period
    min_historical_lookback = historical_days_analysis + sma_period

    # The latest possible start date for a backtest point is `full_historical_data.index.max()` - `simulation_days`
    max_backtest_start_date = full_historical_data.index.max() - timedelta(days=simulation_days) # Approximate - will adjust to actual trading day

    # Filter backtest_data_window index to get valid start dates
    # A date is a valid backtest start date if:
    # 1. It is within the `backtest_days_count` window ending `simulation_days` before today.
    # 2. There are at least `min_historical_lookback` trading days *before* it in `full_historical_data`.
    # 3. There are at least `simulation_days` trading days *after* it in `full_historical_data`.

    valid_backtest_start_dates = []
    # Get all dates in full_historical_data sorted
    all_dates_sorted = full_historical_data.index.sort_values()

    # Iterate through the potential recent start dates
    # We need to check each date to ensure the lookback and lookforward periods exist as *trading days*
    # Instead of iterating every single day, let's iterate through the last `backtest_days_count` trading days
    # that are early enough to allow for the `simulation_days` lookforward.
    potential_start_dates = all_dates_sorted[all_dates_sorted <= full_historical_data.index.max() - timedelta(days=simulation_days)]
    # Take the last `backtest_days_count` of these potential dates
    backtest_start_dates_candidate = potential_start_dates.tail(backtest_days_count)


    for current_start_date in backtest_start_dates_candidate:
         # Check if there are enough historical days before this date for analysis/SMA
         history_before_date = full_historical_data.loc[:current_start_date].iloc[:-1] # Exclude the start date itself from history count
         if len(history_before_date) < min_historical_lookback:
              # Not enough history before this date, skip it
              continue

         # Check if there are enough future days after this date for the actual price comparison
         future_after_date = full_historical_data.loc[current_start_date:].iloc[1:] # Exclude the start date itself from future count
         if len(future_after_date) < simulation_days:
              # Not enough future data for the simulation period, skip it
              continue

         # If both checks pass, this is a valid backtest start date
         valid_backtest_start_dates.append(current_start_date)

    if not valid_backtest_start_dates:
        st.warning(f"Could not find any valid backtest start dates in the last {backtest_days_count} trading days that also have sufficient historical ({historical_days_analysis + sma_period}) and future ({simulation_days}) data.")
        st.stop()


    total_backtest_points = len(valid_backtest_start_dates)
    total_passes = 0
    backtest_messages = [] # Store messages/results for display


    with st.spinner(f"Running backtest over {total_backtest_points} dates ({num_simulations_backtest} simulations per point)..."):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        for i, current_start_date in enumerate(valid_backtest_start_dates):
            progress_bar.progress((i + 1) / total_backtest_points)
            progress_text.text(f"Processing backtest date {i+1}/{total_backtest_points} ({current_start_date.strftime('%Y-%m-%d')})...")

            # 1. Get the historical data window ending on current_start_date needed for analysis/SMA
            historical_window = full_historical_data.loc[:current_start_date].tail(min_historical_lookback)

            # Re-check length just in case (should be covered by valid_backtest_start_dates logic, but defensive)
            if len(historical_window) < min_historical_lookback:
                 backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Insufficient history for analysis ({len(historical_window)} < {min_historical_lookback})")
                 continue

            # 2. Calculate Historical Stats for this window
            original_mean, biased_mean, volatility, sma_applied, sma_message, calc_error = \
                 calculate_historical_stats(
                     historical_window,
                     historical_days_analysis,
                     sma_period,
                     bias_multiplier,
                     ewma_lambda
                 )

            if calc_error:
                 backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Stats calculation failed - {calc_error}")
                 continue # Skip this date if stats calculation failed

            # Get the start price (last price of the historical_window's analysis period tail)
            start_price_data = historical_window.tail(historical_days_analysis).iloc[-1]
            start_price = float(start_price_data.item()) if np.isfinite(start_price_data) else np.nan

            if not np.isfinite(start_price) or start_price <= 0:
                 backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Invalid start price ({start_price})")
                 continue # Skip if start price is invalid

            # 3. Run Monte Carlo Simulation for this point
            median_prices, mean_prices, std_dev_prices, upper_band_1std, lower_band_1std, upper_band_2std, lower_band_2std, sim_error = \
                run_monte_carlo_simulation(start_price, biased_mean, volatility, simulation_days, num_simulations_backtest)

            if sim_error:
                backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Simulation failed - {sim_error}")
                continue # Skip this date if simulation failed

            # 4. Determine the simulation end date
            # The simulation end date is the trading day that is `simulation_days` after `current_start_date`
            # We need to find this date in the `full_historical_data` index.
            try:
                 # Find the index of current_start_date in all_dates_sorted
                 start_date_index_loc = all_dates_sorted.get_loc(current_start_date)
                 # The end date is simulation_days trading days *after* the start date
                 actual_end_date_for_comparison = all_dates_sorted[start_date_index_loc + simulation_days]

                 if actual_end_date_for_comparison > full_historical_data.index.max():
                      # This case should be caught by the initial valid_backtest_start_dates check, but defensive
                      backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Future comparison date ({actual_end_date_for_comparison.strftime('%Y-%m-%d')}) is beyond available data.")
                      continue

                 # Get the actual price on the comparison date
                 actual_end_price = full_historical_data.loc[actual_end_date_for_comparison].item()
                 if not np.isfinite(actual_end_price):
                      backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Actual price on {actual_end_date_for_comparison.strftime('%Y-%m-%d')} is non-finite.")
                      continue


            except Exception as e:
                 backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Error finding future comparison date or price - {e}")
                 continue # Skip if we can't find the actual future price


            # 5. Get the simulated band endpoints for the final step
            if len(upper_band_1std) > 0 and len(lower_band_1std) > 0:
                 simulated_upper_end = upper_band_1std[-1] if np.isfinite(upper_band_1std[-1]) else np.nan
                 simulated_lower_end = lower_band_1std[-1] if np.isfinite(lower_band_1std[-1]) else np.nan
            else:
                 simulated_upper_end = np.nan
                 simulated_lower_end = np.nan
                 backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Simulated band endpoints are not available.")
                 continue # Skip if endpoints aren't finite


            # 6. Compare actual price to the simulated +/- 1 Std Dev band
            if np.isfinite(simulated_lower_end) and np.isfinite(simulated_upper_end):
                 if actual_end_price >= simulated_lower_end and actual_end_price <= simulated_upper_end:
                      total_passes += 1
                      backtest_messages.append(f"PASS: {current_start_date.strftime('%Y-%m-%d')} -> {actual_end_date_for_comparison.strftime('%Y-%m-%d')}. Actual ${actual_end_price:.2f} is WITHIN [${simulated_lower_end:.2f}, ${simulated_upper_end:.2f}].")
                 else:
                      backtest_messages.append(f"FAIL: {current_start_date.strftime('%Y-%m-%d')} -> {actual_end_date_for_comparison.strftime('%Y-%m-%d')}. Actual ${actual_end_price:.2f} is OUTSIDE [${simulated_lower_end:.2f}, ${simulated_upper_end:.2f}].")
            else:
                 backtest_messages.append(f"Skipping {current_start_date.strftime('%Y-%m-%d')}: Simulated band endpoints were non-finite.")


        # Backtest finished
        progress_text.text("Backtest complete.")
        progress_bar.progress(100)

        pass_ratio = (total_passes / total_backtest_points) * 100 if total_backtest_points > 0 else np.nan

        st.session_state.backtest_results = {
             'total_points': total_backtest_points,
             'total_passes': total_passes,
             'pass_ratio': pass_ratio,
             'messages': backtest_messages, # Store detailed messages
             'backtest_days_count_used': backtest_days_count,
             'sim_days_used': simulation_days,
             'num_simulations_backtest_used': num_simulations_backtest,
             'historical_analysis_days_used': historical_days_analysis,
             'sma_period_used': sma_period,
             'bias_multiplier_used': bias_multiplier,
             'ewma_lambda_used': ewma_lambda,
        }
        #st.success("Backtest completed and results stored.")


# --- Display Risk/Reward and Key Forecasts (outside button block) ---
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results

    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    delta_lower_pct = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio = results.get('risk_reward_ratio', np.nan)
    num_simulations_ran = results.get('num_simulations_ran', 'N/A')


    st.subheader("Main Simulation Forecast Insights (+1/-1 Std Dev Based)")
    st.write(f"*(Based on {num_simulations_ran} runs from {results.get('start_price', 'N/A'):.2f} on {results.get('historical_data_close_analyzed', pd.Series()).index.max().strftime('%Y-%m-%d') if not results.get('historical_data_close_analyzed', pd.Series()).empty else 'N/A'})*")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Expected movement to +1 Std Dev End", value=format_percentage(delta_upper_pct, '.2f', 'N/A'))
    with col2:
        st.metric(label="Expected movement to -1 Std Dev End", value=format_percentage(delta_lower_pct, '.2f', 'N/A'))
    with col3:
        risk_reward_str = "N/A"
        if np.isfinite(risk_reward_ratio):
             if risk_reward_ratio == np.inf: risk_reward_str = "Infinite"
             else: risk_reward_str = f"{risk_reward_ratio:.2f} : 1"
        elif np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct): risk_reward_str = "Undetermined / Favorable Downside"

        st.metric(label="Risk/Reward Ratio (+1 Gain : -1 Loss)", value=risk_reward_str)


# --- Display Plot ---
st.subheader("Price Chart: Historical Data, Median, and Standard Deviation Band")
if full_historical_data is not None and not full_historical_data.empty:
    # We need the length of the analysis period data for the plot title
    historical_data_close_analyzed_latest = full_historical_data.tail(historical_days_analysis)
    historical_analysis_len = len(historical_data_close_analyzed_latest)

    fig = plot_simulation(
        full_historical_data,
        historical_days_to_display,
        st.session_state.simulation_results,
        ticker,
        historical_analysis_len, # Pass analysis length
        ewma_lambda,
        sma_period,
        bias_multiplier
    )
    st.pyplot(fig)
    plt.close(fig)
else:
    if 'full_historical_data' in locals() and (full_historical_data is None or full_historical_data.empty):
         st.error("Cannot display plot because historical data fetching failed.")
    else:
         st.warning("Waiting for historical data...")


# --- Display Backtest Results ---
if st.session_state.backtest_results is not None:
    st.subheader("Backtest Results (+1/-1 Std Dev Band)")
    results = st.session_state.backtest_results
    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    col_bt1.metric("Total Backtest Points", results['total_points'])
    col_bt2.metric("Total Passes", results['total_passes'])
    col_bt3.metric("Pass Ratio", f"{results['pass_ratio']:.2f}%" if np.isfinite(results['pass_ratio']) else "N/A")
    col_bt4.metric("Simulations per Point", results['num_simulations_backtest_used'])

    st.write(f"*(Backtest period: last {results['backtest_days_count_used']} trading days, using {results['historical_analysis_days_used']}-day analysis, {results['sma_period_used']}-day SMA bias ({results['bias_multiplier_used']} multiplier), EWMA λ={results['ewma_lambda_used']}, {results['sim_days_used']}-day forecast)*")

    # Optional: Display backtest messages (can be long)
    if st.checkbox("Show Backtest Details"):
         for msg in results['messages']:
              st.write(msg)


# --- Display Results Table ---
if st.session_state.simulation_results is not None:
    st.subheader("Simulation and Analysis Summary")

    results = st.session_state.simulation_results

    historical_data_analyzed = results.get('historical_data_close_analyzed')
    hist_analysis_days = len(historical_data_analyzed) if isinstance(historical_data_analyzed, pd.Series) else 0
    original_mean_log_return = results.get('original_mean_float', np.nan)
    biased_mean_log_return_used = results.get('biased_mean_float_used', np.nan)
    sma_period_used = results.get('sma_period_used', 'N/A')
    bias_multiplier_used = results.get('bias_multiplier_used', 'N/A')
    hist_volatility_log = results.get('volatility_float', np.nan)
    ewma_lambda_used = results.get('ewma_lambda_used', 'N/A')

    last_historical_price_scalar = results.get('start_price', np.nan) # Get from stored results
    last_historical_date_analysis = historical_data_analyzed.index.max() if isinstance(historical_data_analyzed, pd.Series) and not historical_data_analyzed.empty else "N/A"

    num_sims_ran = results.get('num_simulations_ran', 'N/A')
    sim_dates_pd = results.get('simulated_dates', pd.DatetimeIndex([]))
    sim_days = len(sim_dates_pd)
    sim_end_date = sim_dates_pd.max() if len(sim_dates_pd) > 0 else "N/A"

    median_prices_array = results.get('median_prices', np.array([]))
    mean_prices_array = results.get('mean_prices', np.array([]))
    std_dev_prices_array = results.get('std_dev_prices', np.array([]))
    upper_band_array = results.get('upper_band', np.array([]))
    lower_band_array = results.get('lower_band', np.array([]))
    upper_band_2std_array = results.get('upper_band_2std', np.array([]))
    lower_band_2std_array = results.get('lower_band_2std', np.array([]))
    #final_prices_list = results.get('final_prices', []) # Not stored in session_state anymore for space

    # Safely get final values from aggregate arrays
    median_end_price = median_prices_array[-1] if len(median_prices_array) > 0 and np.isfinite(median_prices_array[-1]) else np.nan
    mean_end_price = mean_prices_array[-1] if len(mean_prices_array) > 0 and np.isfinite(mean_prices_array[-1]) else np.nan
    std_dev_end = std_dev_prices_array[-1] if len(std_dev_prices_array) > 0 and np.isfinite(std_dev_prices_array[-1]) else np.nan
    upper_band_end_price_1std = upper_band_array[-1] if len(upper_band_array) > 0 and np.isfinite(upper_band_array[-1]) else np.nan
    lower_band_end_price_1std = lower_band_array[-1] if len(lower_band_array) > 0 and np.isfinite(lower_band_array[-1]) else np.nan
    upper_band_end_price_2std = upper_band_2std_array[-1] if len(upper_band_2std_array) > 0 and np.isfinite(upper_band_2std_array[-1]) else np.nan
    lower_band_end_price_2std = lower_band_2std_array[-1] if len(lower_band_2std_array) > 0 and np.isfinite(lower_band_2std_array[-1]) else np.nan

    # Actual min/max final prices are not stored, remove from table or compute if needed
    actual_min_end_price = "N/A" # Removed
    actual_max_end_price = "N/A" # Removed

    # Get delta percentages and risk/reward from stored values (+/- 1 std dev based)
    delta_upper_pct_1std = results.get('delta_upper_pct', np.nan)
    delta_lower_pct_1std = results.get('delta_lower_pct', np.nan)
    risk_reward_ratio_val = results.get('risk_reward_ratio', np.nan)

    # Recalculate delta percentages for +/- 2 Std Dev FOR TABLE DISPLAY
    delta_upper_pct_2std = np.nan
    delta_lower_pct_2std = np.nan
    if np.isfinite(upper_band_end_price_2std) and np.isfinite(last_historical_price_scalar) and last_historical_price_scalar > 0:
         delta_upper_pct_2std = ((upper_band_end_price_2std - last_historical_price_scalar) / last_historical_price_scalar) * 100
    if np.isfinite(lower_band_end_price_2std) and np.isfinite(last_historical_price_scalar) and last_historical_price_scalar > 0:
         delta_lower_pct_2std = ((lower_band_end_price_2std - last_historical_price_scalar) / last_historical_price_scalar) * 100


    table_data = {
        'Metric': [
            'Historical Analysis Period (days)',
            'Historical Mean Daily Log Return (Original)',
            f'EWMA Daily Log Volatility ($\lambda$={ewma_lambda_used})',
            'Last Historical Price ($)',
            'Last Historical Date',
            '--- Bias Settings Used ---',
            f'SMA Period for Bias ({sma_period_used} days)',
            f'Bias Multiplier ({bias_multiplier_used})',
            f'Biased Mean Daily Log Return (Used in Sim)',
            '--- Simulation Results ---',
            'Number of Simulations',
            f'Simulation Period (days from {(last_historical_date_analysis.strftime("%Y-%m-%d") if isinstance(last_historical_date_analysis, (datetime, pd.Timestamp)) else str(last_historical_date_analysis))})',
            f'Simulation End Date',
            'Simulated Median Ending Price ($)',
            'Simulated Mean Ending Price ($)',
            'Simulated Std Dev Ending Price ($)',
            'Simulated +1 Std Dev Ending Price ($)',
            'Simulated -1 Std Dev Ending Price ($)',
            'Simulated +2 Std Dev Ending Price ($)',
            'Simulated -2 Std Dev Ending Price ($)',
            #'Actual Min Simulated Ending Price ($)', # Removed
            #'Actual Max Simulated Ending Price ($)', # Removed
            'Expected movement to +1 Std Dev End (%)',
            'Expected movement to -1 Std Dev End (%)',
            'Expected movement to +2 Std Dev End (%)',
            'Expected movement to -2 Std Dev End (%)',
            'Risk/Reward Ratio (+1 Gain : -1 Loss)',
        ],
        'Value': [
            hist_analysis_days,
            format_value(original_mean_log_return, ".6f"),
            format_value(hist_volatility_log, ".6f"),
            format_value(last_historical_price_scalar, ".2f"),
            last_historical_date_analysis.strftime('%Y-%m-%d') if isinstance(last_historical_date_analysis, (datetime, pd.Timestamp)) else str(last_historical_date_analysis),
            '',
            sma_period_used,
            bias_multiplier_used,
            format_value(biased_mean_log_return_used, ".6f"),
            '',
            num_sims_ran,
            sim_days,
            sim_end_date.strftime('%Y-%m-%d') if isinstance(sim_end_date, (datetime, pd.Timestamp)) else str(sim_end_date),
            format_value(median_end_price, ".2f"),
            format_value(mean_end_price, ".2f"),
            format_value(std_dev_end, ".2f"),
            format_value(upper_band_end_price_1std, ".2f"),
            format_value(lower_band_end_price_1std, ".2f"),
            format_value(upper_band_end_price_2std, ".2f"),
            format_value(lower_band_end_price_2std, ".2f"),
            #actual_min_end_price, # Removed
            #actual_max_end_price, # Removed
            format_percentage(delta_upper_pct_1std, ".2f"),
            format_percentage(delta_lower_pct_1std, ".2f"),
            format_percentage(delta_upper_pct_2std, ".2f"),
            format_percentage(delta_lower_pct_2std, ".2f"),
            f"{format_value(risk_reward_ratio_val, '.2f')} : 1" if np.isfinite(risk_reward_ratio_val) and risk_reward_ratio_val != np.inf else ("Infinite" if risk_reward_ratio_val == np.inf else "N/A"),
        ]
    }

    results_df = pd.DataFrame(table_data)
    st.dataframe(results_df, hide_index=True, use_container_width=True)
