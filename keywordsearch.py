import streamlit as st
import pandas as pd
import re

def load_data():
    data = pd.read_csv("fashion.csv")
    return data

def search_products(data, query):
    # Define a case-insensitive regex pattern for the search query
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    
    # Apply the pattern to the 'name' column to find matches
    data['Match'] = data['name'].apply(lambda x: bool(pattern.search(x)))
    
    # Filter the data where there is a match
    results = data[data['Match']]
    
    # Limit to top 6 results
    results = results.head(6)
    return results

def highlight_keywords(text, query):
    # Define a case-insensitive regex pattern for the search query
    pattern = re.compile(re.escape(query), re.IGNORECASE)
    
    # Replace the matched text with highlighted HTML (green color)
    highlighted_text = pattern.sub(lambda m: f"<span style='color: red;'>{m.group(0)}</span>", text)
    return highlighted_text

st.title("Fashion Products Keyword Search")

data = load_data()

search_query = st.text_input("Enter a keyword to search for fashion products:")

if search_query:
    search_results = search_products(data, search_query)
    
    st.write(f"Found {len(search_results)} results for '{search_query}':")
    
    for index, row in search_results.iterrows():
        cols = st.columns(3)
        with cols[index % 3]:
            st.image(row['img'], caption=row['name'])
            
            # Highlight keywords only in the title
            highlighted_name = highlight_keywords(row['name'], search_query)
            st.markdown(f"**Product Title:** {highlighted_name}", unsafe_allow_html=True)
            
            st.write(f"Price: {row['price']}")
            st.write(f"Brand: {row['brand']}")
            st.write(f"Description: {row['description']}")
            st.write("---")
