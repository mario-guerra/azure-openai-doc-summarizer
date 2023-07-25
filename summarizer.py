# Import necessary libraries and modules
import argparse
import asyncio
import os
import re
import tiktoken
import PyPDF2
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings

# Dictionary containing the maximum token limits for different GPT models
model_max_tokens = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-35-turbo": 4096,
}

# Default prompt for summarizing the text
default_prompt = """Summarize concisely, highlighting key details, and update with new info. Ignore irrelevant content. Use [PREVIOUS_SUMMARY] and [CURRENT_CHUNK]. Keep first two paragraphs in [PREVIOUS_SUMMARY] as-is. Exclude actual labels from summary. If no content in [PREVIOUS_SUMMARY], list attendees for chats, title for documents. Ensure readability using paragraph breaks."""

# Initialize the semantic kernel
kernel = sk.Kernel()

# Get deployment, API key, and endpoint from environment variables
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

# Initialize the summary service with the deployment, endpoint, and API key
summary_service = AzureChatCompletion(deployment, endpoint, api_key)

# Function to create a summary using the Azure Chat Completion service


async def create_summary(input):
    messages = [("system", default_prompt), ("user", input)]
    reply = await summary_service.complete_chat_async(messages=messages, request_settings=ChatRequestSettings(temperature=0.7, top_p=0.8, max_tokens=2000))
    return(reply)

# Get the encoding model for token estimation
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

# Function to estimate the token count of a given text


def estimate_token_count(text):
    tokens = encoding.encode(text)
    length = len(tokens)
    return length


# Calculate the maximum tokens that can be processed
max_tokens = int(
    (model_max_tokens[deployment] - estimate_token_count(default_prompt)) / 3)

# Set the chunk size for processing text
chunk_size = min(max_tokens * 4, 10000)

# Function to extract text from a PDF file


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

# Function to process text and handle rate limit errors with retries


async def process_text(input_text):
    MAX_RETRIES = 3
    retry_count = 0

    while retry_count < MAX_RETRIES:
        summary = await create_summary(input_text)
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

    raise Exception("Rate limit error. All retries failed.")

# Function to write paragraphs to the output file


def write_paragraphs(out_f, paragraphs):
    for p in paragraphs:
        out_f.write(p + "\n\n")

# Function to extract summary paragraphs from the summary text


def extract_summary_paragraphs(summary_text):
    paragraphs = str(summary_text).split('\n\n')
    return [p.strip() for p in paragraphs]

# Function to summarize a document asynchronously


async def summarize_document(input_path, output_path):
    max_context_paragraphs = 3
    previous_summary_paragraphs = []

    # Remove the output file if it already exists
    if os.path.exists(output_path):
        os.remove(output_path)

    # Extract text from a PDF file or read text from a text file
    if input_path.lower().endswith(".pdf"):
        input_text = extract_text_from_pdf(input_path)
        total_chars = len(input_text)
    else:
        with open(input_path, "r") as f:
            input_text = f.read()
            total_chars = os.stat(input_path).st_size

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
            input_text_chunk = "\n\n".join(
                previous_summary_paragraphs) + "\n\n" + chunk

            # Process the text chunk and generate a summary
            summary_ctx = await process_text(input_text_chunk)

            summary = str(summary_ctx)

            # Update the previous summary paragraphs based on the new summary
            if summary:
                summary_paragraphs = extract_summary_paragraphs(summary)
                while len(summary_paragraphs) > max_context_paragraphs:
                    out_f.write(summary_paragraphs.pop(0) + "\n\n")
                previous_summary_paragraphs = summary_paragraphs
                print("\nSummary window: \n", summary)
            else:
                print("No summary generated for the current chunk.")

            # Calculate and display the progress of the summarization
            progress = (processed_chars / total_chars) * 100
            print(
                f"\nProgress: {processed_chars}/{total_chars} ({progress:.2f}%)")

        # Write the remaining summary paragraphs to the output file
        write_paragraphs(out_f, previous_summary_paragraphs)

    print("Summary complete!")

# Define command-line argument parser
parser = argparse.ArgumentParser(description="Document Summarizer")
parser.add_argument("input_path", help="Path to the input text file")
parser.add_argument("output_path", help="Path to the output summary file")

# Parse command-line arguments
args = parser.parse_args()

# Run the summarization process
if __name__ == "__main__":
    asyncio.run(summarize_document(
        args.input_path, args.output_path))
