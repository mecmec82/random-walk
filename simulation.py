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
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(num_days * 2.5))

    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.error(f"No data fetched for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            return pd.Series() # Return empty Series

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

        return historical_data_close

    except Exception as e:
        st.error(f"Error fetching data for {ticker} using yfinance: {e}")
        st.error("Please check the ticker symbol and your internet connection.")
        return pd.Series()


# --- Fetch Historical Data (Cached - runs on initial load and input changes) ---
full_historical_data = fetch_historical_data(
    ticker=ticker,
    num_days=historical_days_requested
)

# Select the specific subset for ANALYSIS (the most recent historical_days_requested days)
historical_data_close_analyzed = full_historical_data.tail(historical_days_requested)

# --- Check if we have enough data for analysis (NO st.stop() here) ---
enough_data_for_analysis = True
if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
     st.warning("Not enough historical data available for analysis (need at least 2 days with positive prices). Adjust 'Historical Trading Days for Analysis' or check ticker symbol.")
     enough_data_for_analysis = False


# --- Add Slider for Start Simulation X Days Ago ---
max_offset_days_allowed = max(0, len(historical_data_close_analyzed) - 1)
start_offset_days = st.sidebar.slider(
    "Start Simulation X Days Ago",
    min_value=0,
    value=min(7, max_offset_days_allowed),
    max_value=max_offset_days_allowed,
    step=1,
    help="Start the random walk simulation from a historical point, X trading days before the last historical data point used for analysis. This allows you to visualize which simulated paths the actual price has been following during the last X days."
)


# --- Calculate Historical Returns and Volatility (using EWMA) ---
mean_float = np.nan
volatility_float = np.nan

if enough_data_for_analysis: # Only proceed with calculations if initial data is sufficient
    with st.spinner("Calculating historical statistics (including EWMA volatility)..."):
        log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

        if len(log_returns) < 1:
            st.error(f"Not enough valid historical data ({len(historical_data_close_analyzed)} prices) to calculate returns and volatility after dropping NaNs. Need at least 2 consecutive valid prices.")
            enough_data_for_analysis = False
        else:
            mean_daily_log_return = log_returns.mean()
            ewma_alpha = 1 - ewma_lambda
            try:
                ewma_variance_series = log_returns.astype(float).pow(2).ewm(alpha=ewma_alpha, adjust=False, min_periods=0).mean()
                ewma_daily_log_volatility_scalar = np.sqrt(ewma_variance_series.iloc[-1]).item()
                daily_log_volatility = float(ewma_daily_log_volatility_scalar)

                if isinstance(mean_daily_log_return, pd.Series):
                    mean_float = float(mean_daily_log_return.item())
                else:
                    mean_float = float(mean_daily_log_return)
                volatility_float = float(daily_log_volatility)

                if not np.isfinite(mean_float) or not np.isfinite(volatility_float) or volatility_float <= 0:
                    st.error(f"Could not calculate finite, positive volatility ({volatility_float:.6f}) or finite mean ({mean_float:.6f}) from historical data. Check data or analysis period.")
                    enough_data_for_analysis = False
            except Exception as e:
                st.error(f"Error calculating EWMA volatility or mean: {e}")
                enough_data_for_analysis = False


# --- Add Slider for Displayed Historical Days ---
max_display_days = len(full_historical_data)
default_display_days = min(historical_days_requested, max_display_days)
default_display_days = max(100, default_display_days) if max_display_days >= 100 else max_display_days
default_display_days = max(1, default_display_days)

historical_days_to_display = st.sidebar.slider(
    "Historical Days to Display on Plot",
    min_value=min(1, max_display_days),
    max_value=max_display_days,
    value=default_display_days,
    step=10,
    help="Slide to change the number of historical trading days shown on the chart. Does not affect analysis period or calculated volatility."
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
def plot_simulation(full_historical_data, historical_days_to_display, simulation_results, ticker, ewma_lambda):
    fig, ax = plt.subplots(figsize=(14, 7))

    historical_data_to_plot = full_historical_data.tail(historical_days_to_display)
    historical_dates_to_plot_np = historical_data_to_plot.index.values if not historical_data_to_plot.empty else np.array([])
    historical_prices_to_plot_np = historical_data_to_plot.values if not historical_data_to_plot.empty else np.array([])

    if len(historical_prices_to_plot_np) > 0:
        ax.plot(historical_dates_to_plot_np, historical_prices_to_plot_np, label=f'Historical {ticker} Price ({len(historical_data_to_plot)} days displayed)', color='blue', linewidth=2)
    else:
        st.warning("No historical data available to plot.")

    if simulation_results is not None:
         plot_sim_dates_pd = simulation_results.get('plot_sim_dates')
         median_prices = simulation_results.get('median_prices')
         mean_prices = simulation_results.get('mean_prices')
         std_dev_prices = simulation_results.get('std_dev_prices')
         upper_band = simulation_results.get('upper_band')
         lower_band = simulation_results.get('lower_band')
         upper_band_2std = simulation_results.get('upper_band_2std')
         lower_band_2std = simulation_results.get('lower_band_2std')
         num_simulations_ran = simulation_results.get('num_simulations_ran', 'N/A')
         start_offset_days_used = simulation_results.get('start_offset_days_used', 0)
         hist_data_analyzed_for_plot = simulation_results.get('historical_data_close_analyzed')


         plot_sim_dates_np = plot_sim_dates_pd.values if plot_sim_dates_pd is not None else np.array([])

         if (len(plot_sim_dates_np) > 0 and
             median_prices is not None and len(median_prices) == len(plot_sim_dates_np) and np.isfinite(median_prices).any() and
             mean_prices is not None and len(mean_prices) == len(plot_sim_dates_np) and
             std_dev_prices is not None and len(std_dev_prices) == len(plot_sim_dates_np) and
             upper_band is not None and len(upper_band) == len(plot_sim_dates_np) and
             lower_band is not None and len(lower_band) == len(plot_sim_dates_np) and
             upper_band_2std is not None and len(upper_band_2std) == len(plot_sim_dates_np) and
             lower_band_2std is not None and len(lower_band_2std) == len(plot_sim_dates_np)):

              valid_plot_indices = np.isfinite(median_prices)

              plot_sim_dates_valid = plot_sim_dates_np[valid_plot_indices]
              median_prices_valid = median_prices[valid_plot_indices]
              mean_prices_valid = mean_prices[valid_plot_indices]
              std_dev_prices_valid = std_dev_prices[valid_plot_indices]
              upper_band_valid = upper_band[valid_plot_indices]
              lower_band_valid = lower_band[valid_plot_indices]
              upper_band_2std_valid = upper_band_2std[valid_plot_indices]
              lower_band_2std_valid = lower_band_2std[valid_plot_indices]

              if len(plot_sim_dates_valid) > 0:
                 if np.isfinite(upper_band_2std_valid).all() and np.isfinite(lower_band_2std_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_2std_valid, upper_band_2std_valid, color='gold', alpha=0.2, label='+/- 2 Std Dev Band')
                 else:
                      st.warning("+/- 2 Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")

                 if np.isfinite(upper_band_valid).all() and np.isfinite(lower_band_valid).all():
                      ax.fill_between(plot_sim_dates_valid, lower_band_valid, upper_band_valid, color='darkorange', alpha=0.3, label='+/- 1 Std Dev Band')
                 else:
                      st.warning("+/- 1 Std dev band contains non-finite values where median is finite. Band will not be plotted or may be incomplete.")

                 ax.plot(plot_sim_dates_valid, median_prices_valid, label=f'Median Simulated Price ({num_simulations_ran} runs)', color='red', linestyle='-', linewidth=2)

                 if start_offset_days_used > 0 and hist_data_analyzed_for_plot is not None and not hist_data_analyzed_for_plot.empty:
                     sim_start_idx_in_analyzed = len(hist_data_analyzed_for_plot) - 1 - start_offset_days_used

                     if sim_start_idx_in_analyzed >= 0:
                         # The actual data to plot for the overlap should go up to the *last* historical date
                         actual_overlap_data = hist_data_analyzed_for_plot.iloc[sim_start_idx_in_analyzed:]
                         if not actual_overlap_data.empty:
                             ax.plot(actual_overlap_data.index, actual_overlap_data.values,
                                     label=f'Actual Price During Offset ({start_offset_days_used} days)', color='green', linewidth=2, linestyle='--')

                             ax.axvline(x=actual_overlap_data.index[-1], color='gray', linestyle=':', linewidth=1, label='Forecast Divergence Point')
                         else:
                             st.warning("Actual overlap data segment is empty, cannot plot green line.")
                     else:
                         st.warning("Calculated historical index for green line is out of bounds, cannot plot green line.")


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

    plot_title = f'{ticker} Price: Historical Data ({len(historical_data_to_plot)} days displayed, {len(simulation_results.get("historical_data_close_analyzed", [])) if simulation_results else historical_days_requested} days analyzed) and Random Walk Simulation Aggregates (EWMA $\lambda$={ewma_lambda})'
    ax.set_title(plot_title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.grid(True)
    return fig


# --- Main Execution Flow (runs on every rerun) ---

# --- Button to Run Simulation ---
if st.button("Run Simulation"):
    st.session_state.simulation_results = None

    # Re-check data sufficiency just before running the heavy computation
    if not enough_data_for_analysis:
        st.error("Cannot run simulation: Not enough valid historical data for analysis. Please adjust settings.")
        st.stop() # Stop if fundamental data is missing after button click

    with st.spinner(f"Running {num_simulations} simulations for {simulation_days} future days (starting {start_offset_days} days ago) using EWMA $\lambda$={ewma_lambda}..."):

        sim_start_idx_in_analyzed = len(historical_data_close_analyzed) - 1 - start_offset_days

        if sim_start_idx_in_analyzed < 0:
            st.error(f"Cannot run simulation with 'Start Simulation {start_offset_days} Days Ago'. Only {len(historical_data_close_analyzed)} historical days available in analysis period. Please reduce 'X Days Ago' or increase 'Historical Trading Days for Analysis'.")
            st.session_state.simulation_results = None
            st.stop()

        start_price = float(historical_data_close_analyzed.iloc[sim_start_idx_in_analyzed].item())
        simulation_start_date = historical_data_close_analyzed.index[sim_start_idx_in_analyzed]

        if not np.isfinite(start_price) or start_price <= 0:
             st.error(f"Starting historical price ({start_price}) is not a finite positive number. Cannot start simulation.")
             st.session_state.simulation_results = None
             st.stop()

        loc_sim = mean_float
        scale_sim = volatility_float

        if not np.isfinite(loc_sim) or not np.isfinite(scale_sim) or scale_sim <= 0:
             st.error(f"Calculated historical mean ({loc_sim:.6f}) or EWMA volatility ({scale_sim:.6f}) is not finite or volatility is not positive. Cannot run simulation.")
             st.session_state.simulation_results = None
             st.stop()

        total_forecast_length_in_steps = (start_offset_days + 1) + simulation_days

        simulated_dates_pd = pd.date_range(start=simulation_start_date, periods=total_forecast_length_in_steps, freq='B')

        if len(simulated_dates_pd) != total_forecast_length_in_steps:
             st.warning(f"Could not generate exactly {total_forecast_length_in_steps} business days starting from {simulation_start_date.strftime('%Y-%m-%d')}. Generated {len(simulated_dates_pd)}. Simulation path length will be adjusted.")
        
        sim_path_length = len(simulated_dates_pd)

        if sim_path_length < 2:
            st.error(f"Not enough forecast steps generated ({sim_path_length}) for meaningful simulation. Check simulation days or offset.")
            st.session_state.simulation_results = None
            st.stop()

        all_simulated_log_returns = np.random.normal(
             loc=loc_sim,
             scale=scale_sim,
             size=(num_simulations, sim_path_length - 1)
        )

        start_prices_array = np.full((num_simulations, 1), start_price)
        price_change_factors = np.exp(all_simulated_log_returns)
        cumulative_price_multipliers = np.cumprod(np.concatenate((np.ones((num_simulations, 1)), price_change_factors), axis=1), axis=1)
        all_simulated_paths_np = start_prices_array * cumulative_price_multipliers

        if all_simulated_paths_np.shape[1] != sim_path_length:
             st.error(f"Mismatch between generated simulation path length ({all_simulated_paths_np.shape[1]}) and expected length ({sim_path_length}). This indicates an internal calculation error. Cannot plot simulation.")
             st.session_state.simulation_results = None
             st.stop()

        prices_at_each_step = all_simulated_paths_np.T

        median_prices = np.nanmedian(prices_at_each_step, axis=1)
        mean_prices = np.nanmean(prices_at_each_step, axis=1)
        std_dev_prices = np.nanstd(prices_at_each_step, axis=1)

        upper_band = mean_prices + std_dev_prices
        lower_band = mean_prices - std_dev_prices
        upper_band_2std = mean_prices + 2 * std_dev_prices
        lower_band_2std = mean_prices - 2 * std_dev_prices

        final_prices_list_raw = all_simulated_paths_np[:, -1].tolist()
        final_prices = [price for price in final_prices_list_raw if np.isfinite(price)]

        delta_upper_pct = np.nan
        delta_lower_pct = np.nan
        risk_reward_ratio = np.nan

        last_historical_price_scalar = float(historical_data_close_analyzed.iloc[-1].item())

        final_upper_price_1std = upper_band[-1] if len(upper_band) > 0 and np.isfinite(upper_band[-1]) else np.nan
        final_lower_price_1std = lower_band[-1] if len(lower_band) > 0 and np.isfinite(lower_band[-1]) else np.nan

        if np.isfinite(final_upper_price_1std) and last_historical_price_scalar > 0:
             delta_upper_pct = ((final_upper_price_1std - last_historical_price_scalar) / last_historical_price_scalar) * 100

        if np.isfinite(final_lower_price_1std) and last_historical_price_scalar > 0:
             delta_lower_pct = ((final_lower_price_1std - last_historical_price_scalar) / last_historical_price_scalar) * 100

        if np.isfinite(delta_upper_pct) and np.isfinite(delta_lower_pct):
             potential_reward = delta_upper_pct
             potential_risk_abs = -delta_lower_pct

             if potential_risk_abs > 1e-9:
                  risk_reward_ratio = potential_reward / potential_risk_abs
             elif potential_risk_abs >= -1e-9 and potential_reward > 1e-9:
                  risk_reward_ratio = np.inf


    st.session_state.simulation_results = {
        'historical_data_close_analyzed': historical_data_close_analyzed,
        'mean_float': mean_float,
        'volatility_float': volatility_float,
        'ewma_lambda_used': ewma_lambda,
        'plot_sim_dates': simulated_dates_pd,
        'median_prices': median_prices,
        'mean_prices': mean_prices,
        'std_dev_prices': std_dev_prices,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'upper_band_2std': upper_band_2std,
        'lower_band_2std': lower_band_2std,
        'final_prices': final_prices,
        'simulated_dates': simulated_dates_pd,
        'delta_upper_pct': delta_upper_pct,
        'delta_lower_pct': delta_lower_pct,
        'risk_reward_ratio': risk_reward_ratio,
        'num_simulations_ran': num_simulations,
        'simulation_start_date': simulation_start_date,
        'start_offset_days_used': start_offset_days,
        'total_sim_steps': total_forecast_length_in_steps,
        'future_simulation_days_input': simulation_days
    }


# --- Display Risk/Reward and Key Forecasts ---
if st.session_state.simulation_results is not None:
    results = st.session_state.simulation_results

    delta_upper_pct = results.get('delta_upper_pct', np.nan)
    de
