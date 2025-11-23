import streamlit as st
import seaborn as sns

# Streamlit "Hello World"
st.title("Hello Streamlit!")
st.write("This is your first Streamlit app.")

# 🧩 Student Task 1: Add another line that displays any short message. Use one of the appropriate Streamlit functions.
st.write("Welcome to the Iris Data Explorer!")


# Load example dataset
iris = sns.load_dataset("iris")
st.write("Here's the Iris dataset:", iris.head(4))

# 🧩 Student Task 2: Display only the first 10 rows of the dataset using Streamlit.
#st.write("Here's the Iris dataset:", iris.head(8))


# 🧩 Student Task 3: Add a slider that allows the user to select between 5–30 rows to display from the filtered dataset.
species = st.selectbox("Choose a species:", iris["species"].unique())

filtered_data = iris[iris["species"] == species]

st.write(f"Showing data for: **{species}**")
st.dataframe(filtered_data.head())


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


# 🧩 Student Task 5: Use Tabs to cleanup and organize what we have built so far. Put each section into it's own Tab.

tab1, tab2, tab3 = st.tabs(["Introduction", "Data Viewer", "Scatter Plot"])



tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
    st.header("A cat")
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
with tab2:
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)



