# Document Summarizer with Azure OpenAI

Document Summarizer is a Python script that uses Azure OpenAI GPT models to generate rich summaries of input text files (including meeting transcripts) 
or PDF documents. The script processes the input document in chunks and generates a summary by updating the context with new information. The context is a sliding content window consisting of the most recent paragraphs from the previous summary plus the current input text chunk. 

## Dependencies

- Python 3.6 or later
- semantic_kernel
- PyPDF2
- tiktoken

You can install the required libraries using pip:

pip install PyPDF2 tiktoken semantic_kernel

_Note: The `semantic_kernel` library is not a standard library and might not be available through pip. If so, the package can be found [here](https://aka.ms/sk/pypi)_

## Requirements

An Azure OpenAI subscription is required to run this script, along with a deployment for gpt-35-turbo, gpt-4, or gpt-4-32k. 

The different in models is a tradeoff in performance versus speed. The gpt-4 models produce better summary results, but gpt-35-turbo is faster.

Change the name of '.env.example' to '.env' and add your Azure OpenAI deployment name, API key, and enpoint.

## Overview

The script works by reading the input document in chunks and generating a summary using the chat completion functionality of the GPT models from Azure OpenAI. It takes care of rate limits and retries when necessary.

The input document can be a plain text file or a PDF document. The script automatically detects the input file type and extracts the text accordingly.

## Usage

1. Clone the repository to your local machine.

2. Open a terminal and navigate to the folder containing the script file.

3. Install the dependencies

4. Run the script using the following command:

python summarizer.py <input_path> <output_path>

Replace `<input_path>` with the path to the input text file or PDF document, and `<output_path>` with the path to the output summary file.

Example:

python summarizer.py input.pdf summary.txt

This command will generate a summary of the `input.pdf` document and save it in the `summary.txt` file.