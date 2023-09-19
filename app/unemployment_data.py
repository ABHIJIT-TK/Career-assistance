import pandas as pd
import streamlit as st
import plotly.express as px

# Load the CSV data
def run():
    csv_path = "Unemployment in India.csv"  # Replace with the actual path
    data = pd.read_csv(csv_path)

    # Streamlit app
    st.title("Unemployment and Employment Data Visualization")

    # Sidebar filters
    regions = data['Region'].unique()
    selected_region = st.sidebar.selectbox("Select Region", regions)

    areas = data['Area'].unique()
    selected_area = st.sidebar.selectbox("Select Area", areas)

    # Filter data based on selected region and area
    filtered_data = data[(data['Region'] == selected_region) & (data['Area'] == selected_area)]

    # Create graphs
    st.subheader(f"Unemployment Rate in {selected_region} - {selected_area}")
    fig1 = px.line(filtered_data, x=' Date', y=' Estimated Unemployment Rate (%)', title=f"Unemployment Rate in {selected_region} - {selected_area}")
    st.plotly_chart(fig1)

    st.subheader(f"Employment in {selected_region} - {selected_area}")
    fig2 = px.line(filtered_data, x=' Date', y=' Estimated Employed', title=f"Employment in {selected_region} - {selected_area}", line_shape="linear")
    st.plotly_chart(fig2)

    st.subheader(f"Labour Participation Rate in {selected_region} - {selected_area}")
    fig3 = px.line(filtered_data, x=' Date', y=' Estimated Labour Participation Rate (%)', title=f"Labour Participation Rate in {selected_region} - {selected_area}", line_shape="linear")
    st.plotly_chart(fig3)

if __name__ == "__main__":
    run()
