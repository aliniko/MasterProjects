# 🧩 Student Task 5: Build a mini-dashboard using the crime-incidents data in `dataset.csv`. You have the freedom to decide what your 
# dashboard should include and look like. Minimally you have add the following to your dashboard:
#
# Show a preview of the dataset.
# Provide filters such as crime type, year, location description, and arrest (yes/no).
# Let users choose which columns to visualize.
# Include at least one meaningful visualization (e.g., map, scatter, line, or bar).
# Optionally integrate one of the Map widgets or components to geolocate the crime locations

import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

st.title("Crime Incidents Data Dashboard")
st.write("This dashboard presents an analysis of crime incidents data.")
# Load crime incidents dataset

crime_data = pd.read_csv("D:/MasterProjects/data_analysis_wi25_26/2Exercises/Exercise_5/dataset.csv")
st.write("Here's a preview of the Crime Incidents dataset:", crime_data.head(4))

# Filters

columns = ["Primary Type", "Year", "Location Description",  "Arrest"]

order = [False, False, False, True]

selected_crime_type = crime_data.sort_values( by = columns, ascending=order)

st.write("This dashboard presents an analysis of crime incidents data.", selected_crime_type)

# crime_type = st.selectbox("Select Crime Type:", options=selected_crime_type["Primary Type"].unique())


# Let users choose which columns to visualize.
selected_columns = st.multiselect("Select columns to visualize:", options=crime_data.columns.tolist(), default=["Primary Type", "Year", "Location Description",  "Arrest"])
st.write("You selected the following columns:")
filtered_data = crime_data[selected_columns]
st.dataframe(filtered_data.head())
# Visualization
fig, ax = plt.subplots()
ax.bar(filtered_data["Primary Type"].value_counts().index, filtered_data["Primary Type"].value_counts().values, color="blue")
ax.set_xlabel("Primary Type")
ax.set_ylabel("Count")
ax.set_title("Crime Incidents by Primary Type")
st.pyplot(fig)


# Optionally integrate one of the Map widgets or components to geolocate the crime locations
st.map(crime_data)
# Note: Ensure that the dataset.csv file is in the correct path or provide the full path to the file.