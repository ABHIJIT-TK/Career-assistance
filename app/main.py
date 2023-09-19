import streamlit as st
import stock_prediction
import unemployment_data
import job_demands
import plotly.express as px
# Set the background color to white
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff; /* White background */
        font-family: 'Arial', sans-serif; /* Change font */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply rounded corners to elements
st.markdown(
    """
    <style>
    .stApp {
        border-radius: 15px; /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True
)
def main():
    st.title("Program Selection")

    st.markdown("""
        This is a program that provides insights and predictions related to various aspects:

        - **Stock Prediction:** Predict the stock prices of selected companies.
        - **Unemployment Data:** Visualize and analyze unemployment and employment data.
        - **Job Openings Analysis:** Explore and predict job demands and openings.

        Choose a program from the sidebar to get started!
        """)

    # Create a sidebar with program options
    selected_program = st.sidebar.selectbox("Select a Program", [ "","Unemployment Data", "Stock Prediction", "Job Openings Analysis"])

    # Conditional rendering based on the selected program

    if selected_program == "Unemployment Data":
        unemployment_data.run()

    elif selected_program == "Stock Prediction":
        stock_prediction.run()


    elif selected_program == "Job Openings Analysis":
        job_demands.run()

if __name__ == "__main__":
    main()
