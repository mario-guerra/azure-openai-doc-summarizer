# Doc Summarizer with Azure OpenAI

Doc Summarizer is a Python script that uses Azure OpenAI GPT models to generate rich summaries of text files (including meeting transcripts), PDFs, Word documents, and websites.

The script processes the input document in chunks of 10,000 characters, iterating on the summary in several passes using a sliding content window for the context. I experimented with different chunk sizes and 10k seemed to be a sweet spot that produces good summarizes without sacrificing useful information. A larger context size (20k for example) forces the model to use more brevity to stay within the constraints of the model token limits. 

The sliding content window consists of the most recent paragraphs from the previous summary, plus the current input text chunk. As summaries are produced, the oldest paragraphs in the summary are saved off and removed from the content window prior to the next summary iteration. This approach allows the script to iteratively summarize large text files without overflowing the model's token limits, while still retaining enough context between summarization steps to produce a cohesive summary of the entire document.

## Dependencies

- Python 3.6 or later
- semantic_kernel
- tiktoken
- PyPDF2
- python-docx

You can install the required libraries using pip:

`pip install semantic_kernel tiktoken PyPDF2 python-docx requests beautifulsoup4 lxml`

_Note: The `semantic_kernel` library is not a standard library and might not be available through pip. If so, the package can be found [here](https://aka.ms/sk/pypi)_

## Requirements

An Azure OpenAI subscription is required to run this script, along with a deployment for one of the following models:
- gpt-35-turbo
- gpt-4
- gpt-4-32k

The difference in models are tradeoffs between performance and speed. The gpt-4 model produces the best summary results, but gpt-35-turbo is faster. In testing with technical content, gpt-4 was better than the other two models at analysing and retaining technical aspects of the content. The other two models produce decent summaries, but potentially important details are lost.

## Overview

The script works by reading the input document in chunks and generating a summary using the chat completion functionality of the GPT models from Azure OpenAI. It takes care of rate limits and retries when necessary.

The input document can be a plain text file or a PDF document. The script automatically detects the input file type and extracts the text accordingly.

## Usage

1. Clone the repository to your local machine.

2. Open a terminal and navigate to the folder containing the script file.

3. Install the dependencies outlined above

4. Change the name of '.env.example' to '.env' and add your Azure OpenAI deployment name, API key, and endpoint.
 
5. Run the script using the following command:

python summarizer.py <input_path> <output_path>

Replace `<input_path>` with the path to the input file (text, PDF, Word) or URL, and `<output_path>` with the path to the output summary file.

Examples:

`python summarizer.py input.pdf text_summary.txt`

`python summarizer.py https://example.com/no-one-reads-long-articles-anymore.html url_summary.txt`

These commands will generate a summary of the input document or URL and save it in the `*_summary.txt`* file.
