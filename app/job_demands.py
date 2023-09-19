from os import name

import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.linear_model import LinearRegression

def run():
    # Load CSV data
    @st.cache_resource
    def load_data():
        data = pd.read_csv("ani/Naukri Jobs Data.csv")  # Update with your CSV file path
        return data

    data = load_data()

    # Streamlit app layout
    st.title("Job Demands")

    # Display raw data
    st.subheader("Raw Data")
    st.write(data)

    # Preprocess the data
    data['Dates'] = pd.to_datetime(data['Dates'])
    filtered_data = data[data['Dates'].notnull()]

    # Get unique job roles
    job_roles = filtered_data['job_post'].unique()

    # Select a job role from dropdown with autocomplete
    selected_job_role = st.selectbox("Select a job role:", job_roles, index=0)

    # Filter data for the selected job role
    job_filtered_data = filtered_data[filtered_data['job_post'] == selected_job_role]

    # Get unique companies for the selected job role
    companies = job_filtered_data['company'].unique()

    # Select multiple companies from multiselect dropdown
    selected_companies = st.multiselect("Select companies:", companies, default=[companies[0]])

    for selected_company in selected_companies:
        # Filter data for the selected company
        company_data = job_filtered_data[job_filtered_data['company'] == selected_company]

        if not company_data.empty:
            # Display detailed information about the selected company
            st.subheader(f"Details for {selected_company}")
            company_info = company_data.iloc[0]
            st.info(f"Company Rating: {company_info['company_rating']}")
            st.success(f"Experience Required: {company_info['exp_required']}")
            st.warning(f"Job Location: {company_info['job_location']}")
            st.error(f"Job Description:\n{company_info['job_description']}")
            st.info(f"Required Skills:\n{company_info['required_skills']}")
        else:
            st.warning(f"No data available for {selected_company}.")

    st.subheader("Search for Job Openings")
    search_location = st.text_input("Enter a location to search for job openings:", "")

    if search_location:
        filtered_data = data[data['job_location'].str.contains(search_location, case=False)]
        st.write(f"Job Openings in {search_location}:")
        st.dataframe(filtered_data)

    # Job openings by state
    st.subheader("Job Openings by Places  ")
    state_counts = data['job_location'].value_counts()
    num_categories = 10
    current_index = st.slider("Select starting category", 0, len(state_counts) - num_categories)
    selected_states = state_counts.index[current_index:current_index + num_categories]
    selected_state_counts = state_counts[selected_states]
    fig_state = px.bar(selected_state_counts, x=selected_state_counts.index, y=selected_state_counts.values)
    fig_state.update_layout(
        xaxis_title="Job Location",
        yaxis_title="Number of Openings",
        width=800,  # Adjust the width
        height=500  # Adjust the height
    )
    st.plotly_chart(fig_state)

    st.subheader("Search for Job Openings by Job Post")
    search_job_post = st.text_input("Enter a job post to search for job openings:", "")

    if search_job_post:
        filtered_data = data[data['job_post'].str.contains(search_job_post, case=False)]
        st.write(f"Job Openings for {search_job_post} positions:")
        st.dataframe(filtered_data.drop(columns=['Dates']))

    # Job openings by job_post
    st.subheader("Job Openings by Job Post")
    job_post_counts = data['job_post'].value_counts()[:]
    print(len(job_post_counts))# Limit to top 20 job posts for readability
    fig_job_post = px.bar(job_post_counts, x=job_post_counts.index, y=job_post_counts.values)
    fig_job_post.update_layout(
        xaxis_title="Job Post",
        yaxis_title="Number of Openings",
        width=800,  # Adjust the width
        height=500  # Adjust the height
    )
    st.plotly_chart(fig_job_post)

    # Future openings prediction
    # ... (rest of the code)
    # Preprocess the data
    data['Dates'] = pd.to_datetime(data['Dates'])
    filtered_data = data[data['Dates'].notnull()]

    # Get unique job roles
    job_roles = filtered_data['job_post'].unique()



    # Preprocess the data
    data['Dates'] = pd.to_datetime(data['Dates'])
    filtered_data = data[data['Dates'].notnull()]

    # Get unique job roles
    job_roles = filtered_data['job_post'].unique()

    # Select job role from dropdown
    selected_job_role = st.selectbox("Select a job role:", job_roles)

    # Filter data for the selected job role
    job_filtered_data = filtered_data[filtered_data['job_post'] == selected_job_role]

    # Group data by Date and count openings
    openings_by_date = job_filtered_data.groupby('Dates').size().reset_index(name='Openings')

    # Check if the dataset has enough rows for modeling
    if len(openings_by_date) >= 2:
        # Create a time series model for predictions
        prophet_data = openings_by_date.rename(columns={'Dates': 'ds', 'Openings': 'y'})
        model = Prophet()
        model.fit(prophet_data)

        # Create future dates for prediction (next month)
        future_dates = pd.date_range(start=prophet_data['ds'].max(), periods=90, freq='D')

        # Make predictions
        future_dates_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_dates_df)

        # Display predicted openings for the selected job role
        st.subheader(f"Predicted Future Openings for {selected_job_role} in the next month")
        st.write(forecast[['ds', 'yhat']])

        # Plot predicted openings
        st.subheader("Prediction Graph")
        st.line_chart(forecast.set_index('ds')[['yhat']])
    else:
        st.write(f"Not enough data for {selected_job_role} to make predictions.")
if __name__ == "__main__":
    run()