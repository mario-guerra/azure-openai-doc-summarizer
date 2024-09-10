# Copyright (c) 2024 Mario Guerra
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import asyncio
import os
import re
import requests
import PyPDF2
import docx
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from openai import AzureOpenAI

# Load environment variables from .env file if needed
# from dotenv import load_dotenv
load_dotenv()

# Dictionary defining chunk sizes, which influence verbosity of the chat model output.
summary_levels = {
    "verbose": 20000,
    "concise": 20000,
    "terse": 20000,
    "barney": 5000,
    "transcribe": 10000,
}

# Dictionary defining request token sizes, which influence verbosity of the chat model output.
request_token_sizes = {
    "verbose": 10000,
    "concise": 5000,
    "terse": 1000,
    "barney": 3000,
    "transcribe": 10000,
}

summary_prompts = {
    "verbose": """Summarize verbosely, emphasizing key details, action items, and described goals, while incorporating new information from [CURRENT_CHUNK] into [PREVIOUS_SUMMARY]. Retain the first two paragraphs of [PREVIOUS_SUMMARY]. Remove labels, maintain paragraph breaks for readability, and avoid phrases like 'in conclusion' or 'in summary'. Do not reference 'chunk' or 'chunks' in your summary. Collect all questions that were asked but require further follow up, as well as action items.""",
    "concise": """Summarize concisely, highlighting key details and important points, update with new info. Extract and save all questions that were asked but require further follow up. Use [PREVIOUS_SUMMARY] and [CURRENT_CHUNK]. Keep first two paragraphs in [PREVIOUS_SUMMARY] as-is. Exclude these labels from summary. Ensure readability using paragraph breaks, and avoid phrases like 'in conclusion' or 'in summary'.""",
    "terse": """Summarize tersely for executive action using [PREVIOUS_SUMMARY] and [CURRENT_CHUNK], focusing on key details and technical content. Retain the first two paragraphs of [PREVIOUS_SUMMARY], remove labels, and maintain paragraph breaks for readability. Avoid phrases like 'in conclusion' or 'in summary'.""",
    "barney": """Break the content down Barney style, emphasizing key details and incorporating new information from [CURRENT_CHUNK] into [PREVIOUS_SUMMARY]. Retain the first two paragraphs of [PREVIOUS_SUMMARY]. Remove labels, maintain paragraph breaks for readability, and avoid phrases like 'in conclusion' or 'in summary'.""",
    "transcribe": """Convert the following transcript into a dialogue format, similar to a script in a novel. Please remove filler words like 'uh' and 'umm', lightly edit sentences for clarity and readability, and include all the details discussed in the conversation without abbreviating or summarizing any part of it. Maintain paragraph breaks for readability. Do not summarize or omit any details.""",
}

# Set up Azure OpenAI API key and endpoint
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://cognitiveservices.azure.com/.default"
)

client = AzureOpenAI(
    azure_endpoint=endpoint,
    azure_ad_token_provider=token_provider,
    api_version="2024-05-01-preview",
)

# Define a method for creating a summary asynchronously.
async def create_summary(input, summary_level, custom_prompt):
    token = token_provider()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    data = {
        "model": deployment,
        "messages": [
            {
                "role": "user",
                "content": summary_prompts[summary_level] + " " + custom_prompt + "\n" + input
            }
        ],
        "max_tokens": request_token_sizes[summary_level],
        "temperature": 0.4,
        "top_p": 0.4,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
        "stream": False
    }

    response = requests.post(
        f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version=2024-05-01-preview",
        headers=headers,
        json=data
    )

    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Extract text from a Word document
def extract_text_from_word(doc_path):
    doc = docx.Document(doc_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Extract text from a URL
def extract_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")
    text = soup.get_text(separator="\n")
    return text

# Process text and handle ChatGPT rate limit errors with retries.
async def process_text(input_text, summary_level, custom_prompt):
    MAX_RETRIES = 5
    retry_count = 0
    TIMEOUT_DELAY = 5  # Adjust the delay as needed

    while retry_count < MAX_RETRIES:
        try:
            summary = await create_summary(input_text, summary_level, custom_prompt)
            if "exceeded token rate limit" in str(summary):
                error_message = str(summary)
                delay_str = re.search(r'Please retry after (\d+)', error_message)
                if delay_str:
                    delay = int(delay_str.group(1))
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    retry_count += 1
                else:
                    raise Exception("Unknown error message when processing text.")
            else:
                return summary
        except requests.exceptions.RequestException as e:
            if "Request timed out" in str(e):
                print(f"Timeout error occurred. Retrying in {TIMEOUT_DELAY} seconds...")
                await asyncio.sleep(TIMEOUT_DELAY)
                retry_count += 1
            elif "exceeded token rate limit" in str(e):
                error_message = str(e)
                delay_str = re.search(r'Please retry after (\d+)', error_message)
                if delay_str:
                    delay = int(delay_str.group(1))
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    retry_count += 1
            else:
                raise
    if retry_count == MAX_RETRIES:
        if "Request timed out" in str(e):
            raise Exception("Timeout error. All retries failed.")
        else:
            raise Exception("Rate limit error. All retries failed.")

# Write paragraphs to the output file
def write_paragraphs(out_f, paragraphs):
    for p in paragraphs:
        out_f.write(p + "\n\n")
        out_f.flush()

# Extract summary paragraphs from the summary text
def extract_summary_paragraphs(summary_text):
    paragraphs = str(summary_text).split('\n\n')
    return [p.strip() for p in paragraphs]

# Summarize a document asynchronously
async def summarize_document(input_path, output_path, summary_level):
    max_context_paragraphs = 3
    previous_summary_paragraphs = []

    # Set the chunk size for processing text based on summary level.
    chunk_size = summary_levels[summary_level]

    # Remove the output file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Check the input file type and extract text accordingly
    if input_path.lower().startswith("http"):
        input_text = extract_text_from_url(input_path)
    else:
        file_extension = input_path.lower().split('.')[-1]
        if file_extension == "pdf":
            input_text = extract_text_from_pdf(input_path)
        elif file_extension == "docx":
            input_text = extract_text_from_word(input_path)
        else:
            with open(input_path, "r", encoding="utf-8") as f:
                input_text = f.read()

    total_chars = len(input_text)

    # Process the input text in chunks and generate the summary
    with open(output_path, "a") as out_f:
        processed_chars = 0
        while True:
            print("Processing...")
            # Read a chunk of text from the input_text
            chunk = input_text[processed_chars:processed_chars+chunk_size]
            processed_chars += len(chunk)

            # Break the loop if there's no more text to process
            if not chunk:
                break

            # Combine previous summary paragraphs and the current chunk
            if summary_level == "transcribe":
                input_text_chunk = chunk
            else:
                input_text_chunk = "[PREVIOUS_SUMMARY]\n\n" + "\n\n".join(
                    previous_summary_paragraphs) + "\n\n" + "[CURRENT_CHUNK]\n\n" + chunk

            # Process the text chunk and generate a summary
            summary_ctx = await process_text(input_text_chunk, summary_level, args.prompt)
            summary = str(summary_ctx)

            # Update the previous summary paragraphs based on the new summary.
            if summary:
                summary_paragraphs = extract_summary_paragraphs(summary)
                while len(summary_paragraphs) > max_context_paragraphs:
                    out_f.write(summary_paragraphs.pop(0) + "\n\n")
                    out_f.flush()
                previous_summary_paragraphs = summary_paragraphs
                print("\nSummary window: \n", summary)
            else:
                print("No summary generated for the current chunk.")

            # Calculate and display the progress of the summarization
            progress = (processed_chars / total_chars) * 100
            print(f"\nProgress: {processed_chars}/{total_chars} ({progress:.2f}%)")

        # Write the remaining summary paragraphs to the output file
        while previous_summary_paragraphs:
            out_f.write(previous_summary_paragraphs.pop(0) + "\n\n")
            out_f.flush()
    print("Process complete!")

# Define command-line argument parser
parser = argparse.ArgumentParser(description="Document Summarizer")
parser.add_argument("input_path", help="Path to the input text file")
parser.add_argument("output_path", help="Path to the output summary file")
parser.add_argument("--summary-level", choices=["verbose", "concise", "terse", "barney", "transcribe"],
                    default="verbose", help="Configure summary level, verbose is default")
parser.add_argument("--prompt", default="", help="Add a custom prompt to the instructional prompt")

# Parse command-line arguments
args = parser.parse_args()

# Run the summarization process
if __name__ == "__main__":
    asyncio.run(summarize_document(
        args.input_path, args.output_path, summary_level=args.summary_level))