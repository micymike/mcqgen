from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.callbacks import get_openai_callback
import os
import json
import pandas as pd
import traceback
from dotenv import load_dotenv
import PyPDF2

# Load environment variables
load_dotenv()

# Load OpenAI API key
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise ValueError("OpenAI API key not found. Please set it in the environment variables.")

# Initialize the language model
llm = ChatOpenAI(openai_api_key=key, model_name="gpt-3.5-turbo", temperature=0.7)

# Define RESPONSE_JSON directly in the script
RESPONSE_JSON = {
    "1": {
        "no": "1",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    "2": {
        "no": "2",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    },
    "3": {
        "no": "3",
        "mcq": "multiple choice questions",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here"
        },
        "correct": "correct answer"
    }
}

# Print the loaded JSON
print(RESPONSE_JSON)

# Define prompt templates
TEMPLATE = """
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like RESPONSE_JSON below and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{RESPONSE_JSON}
"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    template=TEMPLATE
)

TEMPLATE2 = """
You are an expert English grammarian and writer. Given a Multiple Choice Quiz for {subject} students,\
you need to evaluate the complexity of the questions and give a complete analysis of the quiz. Only use at most 50 words for complexity analysis. 
If the quiz is not up to par with the cognitive and analytical abilities of the students,\
update the quiz questions that need to be changed and change the tone such that it perfectly fits the students' abilities.
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(
    input_variables=["subject", "quiz"],
    template=TEMPLATE2
)

# Define the LLM Chains
quiz_chain = LLMChain(llm=llm, prompt=quiz_generation_prompt, output_key="quiz", verbose=True)
review_chain = LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)

# Sequential chain to generate and evaluate quiz
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],
    input_variables=["text", "number", "subject", "tone", "RESPONSE_JSON"],
    output_variables=["quiz", "review"],
    verbose=True,
)

# Function to read uploaded file
def read_file(file_path):
    with open(file_path, 'rb') as file:
        if file_path.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
            return text
        else:
            return file.read().decode("utf-8")

# Main function to process the file and generate MCQs
def main(file_path, mcq_count, subject, tone):
    try:
        # Read the uploaded file
        text = read_file(file_path)
        
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
        print(f"An error occurred: {e}")
        traceback.print_exc()
    else:
        print("MCQs generated successfully!")
        print(f"Total Tokens: {cb.total_tokens}")
        print(f"Prompt Tokens: {cb.prompt_tokens}")
        print(f"Completion Tokens: {cb.completion_tokens}")
        print(f"Total Cost: {cb.total_cost}")
        
        if isinstance(response, dict):
            # Extract the quiz data from the response
            quiz = response.get("quiz")
            if quiz is not None:
                table_data = json.loads(quiz)  # Ensure quiz is loaded correctly
                if table_data:
                    df = pd.DataFrame(table_data)
                    df.index = df.index + 1
                    print(df)
                    # Display the review as well
                    print("Review:")
                    print(response.get("review", ""))
                else:
                    print("Error processing the table data.")
            else:
                print("No quiz data found in the response.")
        else:
            print(response)

# Example usage
if __name__ == "__main__":
    file_path = "path_to_your_file.pdf"  # Replace with your file path
    mcq_count = 10  # Number of MCQs to generate
    subject = "Science"  # Subject
    tone = "Medium"  # Complexity level
    main(file_path, mcq_count, subject, tone)
