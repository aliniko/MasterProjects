import streamlit as st
import seaborn as sns

# Streamlit "Hello World"
st.title("Hello Streamlit!")
st.write("This is your first Streamlit app.")

# 🧩 Student Task 1: Add another line that displays any short message. Use one of the appropriate Streamlit functions.


# Load example dataset
iris = sns.load_dataset("iris")
st.write("Here's the Iris dataset:", iris.head(4))

# 🧩 Student Task 2: Display only the first 10 rows of the dataset using Streamlit.
#st.write("Here's the Iris dataset:", iris.head(8))

species = st.selectbox("Choose a species:", iris["species"].unique())

filtered_data = iris[iris["species"] == species]

st.write(f"Showing data for: **{species}**")
st.dataframe(filtered_data.head())

# 🧩 Student Task 3: Add a slider that allows the user to select between 5–30 rows to display from the filtered dataset.

import matplotlib.pyplot as plt

# Create a simple scatter plot
fig, ax = plt.subplots()
ax.scatter(iris["sepal_length"], iris["sepal_width"], color="teal")
ax.set_xlabel("Sepal Length")
ax.set_ylabel("Sepal Width")
ax.set_title("Sepal Length vs Width")
st.pyplot(fig)

# 🧩 Student Task 4: Create a scatter plot for `petal_length` vs `petal_width` instead.

fig, ax = plt.subplots()
ax.scatter(iris["sepal_length"], iris["sepal_width"], color="teal")
ax.set_xlabel("petal length")
ax.set_ylabel("petal width")
ax.set_title("petal_length` vs `petal_width")
st.pyplot(fig)

