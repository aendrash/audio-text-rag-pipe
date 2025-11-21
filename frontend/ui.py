import streamlit as st
import requests

API_URL = st.secrets["API_URL"]


st.title("🎧 AI Audio Search Engine")

query = st.text_input("Describe the sound you want:")

if st.button("Search"):
    if not query.strip():
        st.warning("Enter a query.")
    else:
        response = requests.post(API_URL, json={"query": query}).json()

        for item in response["results"]:
            st.subheader(f"{item['file']} — {item['class']}")
            st.write(item["description"])

            with open(item["path"], "rb") as f:
                st.audio(f.read())
