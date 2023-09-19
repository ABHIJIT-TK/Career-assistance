import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

def run():
    # Load CSV data
    @st.cache_resource
    def load_data():
        data = pd.read_csv("Naukri Jobs Data.csv")  # Update with your CSV file path
        return data

    data = load_data()

    # Streamlit app layout
    st.title("Job Demands")

    # Display raw data
    st.subheader("Raw Data")
    st.write(data)

    st.subheader("Search for Job Openings")
    search_location = st.text_input("Enter a location to search for job openings:", "")

    if search_location:
        filtered_data = data[data['job_location'].str.contains(search_location, case=False)]
        st.write(f"Job Openings in {search_location}:")
        st.dataframe(filtered_data.drop(columns=['Posted_as_on_22_5_2022']))

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
        st.dataframe(filtered_data.drop(columns=['Posted_as_on_22_5_2022']))

    # Job openings by job_post
    st.subheader("Job Openings by Job Post")
    job_post_counts = data['job_post'].value_counts()[:20]  # Limit to top 20 job posts for readability
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

if __name__ == "__main__":
    run()
