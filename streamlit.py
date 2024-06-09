import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import get_openai_callback
from src.mcqgenerator.mcqgenerator import generate_evaluate_chain
from src.mcqgenerator.utils import read_file, get_table_data
from src.mcqgenerator.logger import logging

# Load environment variables
load_dotenv()

# Load JSON response file
with open('Response.json', 'r') as file:
    RESPONSE_JSON = json.load(file)

# Create a title for the app
st.title("Mike's MCQs Creator Application with LangChain 🦜⛓️")

# Form for user inputs
with st.form("user_input"):
    uploaded_file = st.file_uploader("Upload PDF or Text", type=['pdf', 'txt'])
    mcq_count = st.number_input("Number of MCQs", min_value=3, max_value=50)
    subject = st.text_input("Insert Subject", max_chars=20)
    tone = st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")
    button = st.form_submit_button("Create MCQs")

    if button and uploaded_file is not None and mcq_count and subject and tone:
        with st.spinner("Generating MCQs..."):
            try:
                # Read the uploaded file
                text = read_file(uploaded_file)
                
                # Count tokens and the cost of API call
                with get_openai_callback() as cb:
                    response = generate_evaluate_chain({
                        "text": text,
                        "number": mcq_count,
                        "subject": subject,
                        "tone": tone,
                        "RESPONSE_JSON": RESPONSE_JSON  # Pass the loaded JSON directly
                    })
        
            except Exception as e:
                st.error(f"An error occurred: {e}")
                traceback.print_exc()
            else:
                st.success("MCQs generated successfully!")
                st.write(f"Total Tokens: {cb.total_tokens}")
                st.write(f"Prompt Tokens: {cb.prompt_tokens}")
                st.write(f"Completion Tokens: {cb.completion_tokens}")
                st.write(f"Total Cost: {cb.total_cost}")
                
                if isinstance(response, dict):
                    # Extract the quiz data from the response
                    quiz = response.get("quiz")
                    if quiz is not None:
                        table_data = get_table_data(quiz)
                        if table_data is not None:
                            df = pd.DataFrame(table_data)
                            df.index = df.index + 1
                            st.table(df)
                            # Display the review in a text area as well
                            st.text_area(label="Review", value=response.get("review", ""), height=200)
                        else:
                            st.error("Error processing the table data Please try again.")
                    else:
                        st.error("No quiz data found in the response.")
                else:
                    st.write(response)
