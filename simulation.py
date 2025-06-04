import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
import io # For potential CSV reading, though JSON is used

# --- Streamlit App Configuration ---
st.set_page_config(page_title="Stock Random Walk Simulation", layout="wide")

st.title("Stock Price Random Walk Simulation")
st.write("Simulate future stock price movements using a random walk based on historical volatility.")

# --- Input Parameters ---
st.sidebar.header("Simulation Settings")

api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
ticker = st.sidebar.text_input("Stock Ticker Symbol (e.g., SPY)", 'SPY').upper()
historical_days = st.sidebar.number_input("Historical Trading Days for Analysis", min_value=50, value=300, step=10)
simulation_days = st.sidebar.number_input("Future Simulation Days", min_value=1, value=30, step=1)

st.sidebar.write("Note: Alpha Vantage free tier has API request limits.")

# --- Main Simulation Logic (triggered by button) ---
if st.button("Run Simulation"):
    if not api_key:
        st.error("Please enter your Alpha Vantage API Key in the sidebar.")
    else:
        with st.spinner(f"Fetching historical data for {ticker}..."):
            # --- Fetch Historical Data from Alpha Vantage ---
            AV_FUNCTION = 'TIME_SERIES_DAILY_ADJUSTED'
            AV_OUTPUT_SIZE = 'full' # Use 'full' to get more than the last 100 days
            url = f'https://www.alphavantage.co/query?function={AV_FUNCTION}&symbol={ticker}&outputsize={AV_OUTPUT_SIZE}&apikey={api_key}'

            try:
                response = requests.get(url)
                response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
                data = response.json()

                # Check if the expected data key exists
                time_series_key = "Time Series (Daily)"
                if time_series_key not in data:
                    error_message = "Could not retrieve time series data from Alpha Vantage."
                    if "Note" in data:
                         error_message += f"\nAPI Note: {data['Note']}"
                    elif "Error Message" in data:
                         error_message += f"\nAPI Error: {data['Error Message']}"
                    st.error(error_message)
                    st.json(data) # Show the response data for debugging
                    st.stop() # Stop the rest of the execution

                raw_dates = []
                raw_prices = []

                # Alpha Vantage returns data in reverse chronological order (most recent first)
                # We need to parse it and store it in a way we can sort by date.
                # Also, filter for '4. adjusted close'
                for date_str, daily_data in data[time_series_key].items():
                    try:
                        price = float(daily_data['4. adjusted close'])
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        raw_dates.append(date_obj)
                        raw_prices.append(price)
                    except (ValueError, KeyError) as e:
                        st.warning(f"Could not parse data for date {date_str}. Error: {e}")
                        continue # Skip this date if parsing fails

                # Create a pandas Series and sort by date (ascending)
                historical_data_close = pd.Series(raw_prices, index=raw_dates)
                historical_data_close = historical_data_close.sort_index()

                # Ensure we have enough data and take the last 'historical_days' trading days
                if len(historical_data_close) < historical_days:
                    st.warning(f"Only {len(historical_data_close)} days of data available from Alpha Vantage for {ticker}. Using all available data ({len(historical_data_close)} days) for historical analysis.")
                    # Use all available data if less than requested historical_days
                    historical_data_close_analyzed = historical_data_close
                else:
                    historical_data_close_analyzed = historical_data_close.tail(historical_days)

                if historical_data_close_analyzed.empty or len(historical_data_close_analyzed) < 2:
                     st.error(f"Not enough historical data ({len(historical_data_close_analyzed)} days) available or parsed for analysis. Need at least 2 days.")
                     st.stop()

            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching data from Alpha Vantage API: {e}")
                st.error("Please check your internet connection and API key.")
                st.stop() # Stop execution if API call fails
            except Exception as e:
                st.error(f"An unexpected error occurred during data processing: {e}")
                st.stop() # Stop execution for other errors

        # --- Calculate Historical Returns and Volatility ---
        with st.spinner("Calculating historical statistics..."):
            log_returns = np.log(historical_data_close_analyzed / historical_data_close_analyzed.shift(1)).dropna()

            mean_daily_log_return = log_returns.mean()
            daily_log_volatility = log_returns.std()

            st.subheader("Historical Analysis Results")
            st.write(f"Based on the last **{len(historical_data_close_analyzed)}** trading days of **{ticker}**:")
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
        last_historical_date = historical_dates[-1]

        # Generate future dates for the simulation, starting after the last historical date
        simulated_dates = pd.date_range(start=last_historical_date, periods=simulation_days + 1, freq='B')[1:] # [1:] to exclude the start date itself

        # Ensure simulated dates match simulation steps
        if len(simulated_dates) != simulation_days:
             st.warning(f"Could not generate {simulation_days} business days after {last_historical_date}. Generated {len(simulated_dates)}. Trimming simulation results.")
             # Trim simulation data if dates couldn't be generated
             simulated_price_path = simulated_price_path[:len(simulated_dates) + 1]


        # --- Plotting ---
        st.subheader("Price Chart: Historical and Simulated")

        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot Historical Data
        ax.plot(historical_dates, historical_data_close_analyzed, label=f'Historical {ticker} Price', color='blue')

        # Plot Simulated Data
        # The x-axis for the simulated path includes the last historical date + the future simulated dates
        plot_sim_dates = np.concatenate(([last_historical_date], simulated_dates))

        if len(plot_sim_dates) == len(simulated_price_path):
             ax.plot(plot_sim_dates, simulated_price_path, label=f'Simulated Future Price ({len(simulated_dates)} days)', color='red', linestyle='--')
        else:
             st.error("Error plotting simulation: Date and price length mismatch.")
             st.write(f"Plot dates length: {len(plot_sim_dates)}, Simulated prices length: {len(simulated_price_path)}")


        ax.set_title(f'{ticker} Price: Historical Data and Random Walk Simulation')
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
        if len(simulated_dates) > 0:
             st.write(f"**Simulated Price** after {len(simulated_dates)} steps ({simulated_dates[-1].strftime('%Y-%m-%d')}): **${simulated_price_path[-1]:.2f}**")
        else:
             st.warning("No simulated dates or prices were generated.")
