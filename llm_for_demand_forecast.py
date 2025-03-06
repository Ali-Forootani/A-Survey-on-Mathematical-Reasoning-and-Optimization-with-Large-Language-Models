import pandas as pd
import numpy as np
from openai import OpenAI

import matplotlib.pyplot as plt


# Initialize OpenAI API
def initialize_openai(api_key, api_base=None, api_version=None):
    client = OpenAI(api_key=api_key, base_url=api_base)
    return client

# Generate Mock Energy Consumption Data
def generate_mock_data(start_date, periods, freq='h'):
    """
    Generate a mock dataset for energy consumption.
    """
    date_range = pd.date_range(start=start_date, periods=periods, freq=freq)
    energy_consumption = np.random.uniform(100, 500, size=periods)  # Mock energy values
    return pd.DataFrame({'timestamp': date_range, 'consumption': energy_consumption})

# Summarize historical and real-time data
def summarize_data(df):
    """
    Summarize historical trends from the dataset.
    """
    summary = {
        "average_consumption": df['consumption'].mean(),
        "max_consumption": df['consumption'].max(),
        "min_consumption": df['consumption'].min(),
        "latest_consumption": df['consumption'].iloc[-1],
        "trend": "increasing" if df['consumption'].iloc[-1] > df['consumption'].iloc[-24:].mean() else "decreasing"
    }
    return summary

# Forecast function using OpenAI client
def query_llm_for_forecast(client, summary, historical_data):
    """
    Query the LLM to forecast demand based on summarized data and historical trends.
    """
    prompt = f"""
You are an expert in energy demand forecasting and optimization. Based on the following summarized historical energy consumption data, forecast the energy demand for the next hour and suggest optimization strategies.

Summary of Data:
- Average Consumption: {summary['average_consumption']:.2f} kWh
- Maximum Consumption: {summary['max_consumption']:.2f} kWh
- Minimum Consumption: {summary['min_consumption']:.2f} kWh
- Latest Consumption: {summary['latest_consumption']:.2f} kWh
- Trend: {summary['trend']}

Recent Readings:
{historical_data.tail(10).to_string(index=False)}

Provide a clear forecast and actionable optimization strategies.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust if required by your deployment
            messages=[
                {"role": "system", "content": "You are an energy demand forecasting expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=800
        )
        report = response.choices[0].message.content
        return report
    except Exception as e:
        return f"Error generating forecast report: {e}"

# Main execution
if __name__ == "__main__":
    # Replace with your API key and base URL
    api_key = "Your API Key"  # Replace with your actual API key
    api_base = "Your API Base"  # Replace with your base URL
    api_version = "2025-01-01"  # Optional, based on your deployment

    # Initialize OpenAI API
    client = initialize_openai(api_key, api_base, api_version)

    print("Generating Energy Demand Forecast and Optimization Report...\n")

    # Generate mock data
    historical_data = generate_mock_data(start_date="2025-01-01 00:00", periods=100)
    
    # Summarize data
    summary = summarize_data(historical_data)

    # Query the LLM
    report = query_llm_for_forecast(client, summary, historical_data)
    
    # Output report
    print(report)
    
    
    
    # Plot the mock energy consumption data
    plt.figure(figsize=(12, 6))
    plt.plot(historical_data['timestamp'], historical_data['consumption'], marker='o', linestyle='-', label='Energy Consumption')
    plt.title('Mock Energy Consumption Data', fontsize=16)
    plt.xlabel('Timestamp', fontsize=14)
    plt.ylabel('Consumption (kWh)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save the plot as an image file (e.g., PNG, JPEG, PDF)
    plt.savefig("mock_energy_consumption_plot.png", dpi=300, format='png')  # Change format if needed
    plt.show()
    
    
    
    
    
    
