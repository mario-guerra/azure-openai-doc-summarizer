import argparse
import asyncio
import os
import re
import tiktoken
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings
from semantic_kernel.connectors.ai.ai_exception import AIException

# Dictionary defining chunk sizes, which influence verbosity of the chat model output.
# The larger the chunk size, the more verbose the output. The chunk size is
# used to determine the number of characters to process in a given text during a
# single request to the chat model.
summary_levels = {
    "verbose": 5000,
    "concise": 10000,
    "terse": 20000,
}

# Dictionary defining request token sizes, which influence verbosity of the chat model output.
# The larger the request token size, the more verbose the output. The request token size is
# used to determine the number of tokens to request from the chat model during a single request.
request_token_sizes = {
    "verbose": 3000,
    "concise": 2000,
    "terse": 1000,
}

summary_prompts = {
    "verbose": """Summarize verbosely, emphasizing key details and incorporating new information from [CURRENT_CHUNK] into [PREVIOUS_SUMMARY]. Retain the first two paragraphs of [PREVIOUS_SUMMARY]. Remove labels, maintain paragraph breaks for readability, and avoid phrases like 'in conclusion' or 'in summary'.""",
    "concise": """Summarize concisely, highlighting key details, and update with new info. Ignore irrelevant content, include all technical content. Use [PREVIOUS_SUMMARY] and [CURRENT_CHUNK]. Keep first two paragraphs in [PREVIOUS_SUMMARY] as-is. Exclude these labels from summary. Ensure readability using paragraph breaks, and avoid phrases like 'in conclusion' or 'in summary'.""",
    "terse": """Summarize tersely for executive action using [PREVIOUS_SUMMARY] and [CURRENT_CHUNK], focusing on key details and technical content. Retain the first two paragraphs of [PREVIOUS_SUMMARY], remove labels, and maintain paragraph breaks for readability. Avoid phrases like 'in conclusion' or 'in summary'.""",
}

# Initialize the semantic kernel for use in getting settings from .env file.
# I'm not using the semantic kernel pipeline for communicating with the GPT models,
# I'm using the semantic kernel service connectors directly for simplicity.
kernel = sk.Kernel()

# Get deployment, API key, and endpoint from environment variables
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

# Using the chat completion service for summarizing text. 
# Initialize the summary service with the deployment, endpoint, and API key
summary_service = AzureChatCompletion(deployment, endpoint, api_key)

# Get the encoding model for token estimation. This is needed to estimate the number of tokens the
# text chunk will take up, so that we can process the text in chunks that fit within the context window.
# gpt-3.5-turbo uses the same encoding model as gpt-4, so we can use the same encoding model for token estimation.
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Estimate the token count of a given text
def estimate_token_count(text):
    tokens = encoding.encode(text)
    length = len(tokens)
    return length

# Define a method for creating a summary asynchronously. Each time this method is called,
# a list of messages is created and seeded with the system prompt, along with the user input.
# The user input consists of a portion of the previous summary, along with the current text chunk
# being processed.
#
# The number of tokens requested from the model is based on the tokenized size of the
# input text plus the system prompt tokens. The larger the chunk size, the fewer tokens
# we can request from the model to fit within the context window. Therefore the model
# will be less verbose with larger chunk sizes.
async def create_summary(input, summary_level):
    messages = [("system", summary_prompts[summary_level]), ("user", input)]
    request_size = request_token_sizes[summary_level]
    reply = await summary_service.complete_chat_async(messages=messages,request_settings=ChatRequestSettings(temperature=0.4, top_p=0.4, max_tokens=request_size))
    return(reply)

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

# Process text and handle ChatGPT rate limit errors with retries. Rate limit errors
# are passed as a string in the summary text rather than thrown as an exception, which
# is why we need to check for the error message in the summary text. If a rate limit
# error is encountered, the method will retry the request after the specified delay.
# The delay is extracted from the error message, since it explicitly states how long
#  to wait before a retry.
async def process_text(input_text, summary_level):
    MAX_RETRIES = 5
    retry_count = 0
    TIMEOUT_DELAY = 5  # Adjust the delay as needed

    request_size = request_token_sizes[summary_level]

    while retry_count < MAX_RETRIES:
        try:
            summary = await create_summary(input_text, summary_level)
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
        except AIException as e:
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
            with open(input_path, "r") as f:
                input_text = f.read()

    total_chars = len(input_text)

    # Process the input text in chunks and generate the summary
    with open(output_path, "a") as out_f:
        processed_chars = 0
        while True:
            print("Summarizing...")
            # Read a chunk of text from the input_text
            chunk = input_text[processed_chars:processed_chars+chunk_size]
            processed_chars += len(chunk)

            # Break the loop if there's no more text to process
            if not chunk:
                break

            # Combine previous summary paragraphs and the current chunk
            input_text_chunk = "[PREVIOUS_SUMMARY]\n\n" + "\n\n".join(
                previous_summary_paragraphs) + "\n\n" + "[CURRENT_CHUNK]\n\n" + chunk

            # Process the text chunk and generate a summary
            summary_ctx = await process_text(input_text_chunk, summary_level)

            summary = str(summary_ctx)

            # Update the previous summary paragraphs based on the new summary.
            # If the summary has more than max_context_paragraphs, remove the first
            # paragraph until the summary is within the limit. As paragraphs are removed,
            # they are written to the output file.
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
            print(
                f"\nProgress: {processed_chars}/{total_chars} ({progress:.2f}%)")

        # Write the remaining summary paragraphs to the output file
        # write_paragraphs(out_f, previous_summary_paragraphs)
        while previous_summary_paragraphs:
            out_f.write(previous_summary_paragraphs.pop(0) + "\n\n")
            out_f.flush()
    print("Summary complete!")

# Define command-line argument parser
parser = argparse.ArgumentParser(description="Document Summarizer")
parser.add_argument("input_path", help="Path to the input text file")
parser.add_argument("output_path", help="Path to the output summary file")
parser.add_argument("--summary-level", choices=["verbose", "concise", "terse"],
                    default="concise", help="Configure summary level, concise is default")

# Parse command-line arguments
args = parser.parse_args()

# Run the summarization process
if __name__ == "__main__":

    asyncio.run(summarize_document(
        args.input_path, args.output_path, summary_level=args.summary_level))
