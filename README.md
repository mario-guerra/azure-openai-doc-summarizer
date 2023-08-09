# Doc Summarizer with Azure OpenAI

Doc Summarizer is a Python script that uses Azure OpenAI models to generate rich summaries of large documents, including text files, PDFs, Word documents, and websites.

The script processes the input document in chunks based on the selected summary level (verbose, concise, or terse), iterating on the summary using a sliding content window for the context. 

The sliding content window consists of the most recent paragraphs from the previous summary, plus the current input text chunk. This approach allows the script to iteratively summarize large text files without overflowing the model's token limits, while still retaining enough context between summarization steps to produce a cohesive summary of the entire document.

## Dependencies

- Python 3.6 or later
- semantic_kernel
- tiktoken
- PyPDF2
- python-docx
- requests
- beautifulsoup4
- lxml

You can install the required libraries using pip:

`pip install semantic_kernel tiktoken PyPDF2 python-docx requests beautifulsoup4 lxml`

_Note: The `semantic_kernel` library is not a standard library and might not be available through pip. If so, the package can be found [here](https://aka.ms/sk/pypi)_

## Requirements

An Azure OpenAI subscription is required to run this script, along with a deployment for one of the following models:
- gpt-3.5-turbo
- gpt-4
- gpt-4-32k

The difference in models are tradeoffs between performance and speed. The gpt-4 model produces the best summary results, but gpt-35-turbo is faster. In testing with technical content, gpt-4 was better than the other two models at analysing and retaining technical aspects of the content. The other two models produce decent summaries, but potentially important details are lost.

## Overview

The script reads the input document in chunks and uses the `AzureChatCompletion` connector from the [Semantic Kernel library](https://github.com/microsoft/semantic-kernel) to generate summaries. Token rate limit errors and timeouts are handled with retry logic. The final summary is written to the specified output file.

The input document can be a plain text file, a PDF document, a Word document, or a URL. The script automatically detects the input file type and extracts the text accordingly.

## Usage

1. Clone the repository to your local machine.

2. Open a terminal and navigate to the folder containing the script file.

3. Install the dependencies outlined above using pip: `pip install semantic_kernel tiktoken PyPDF2 python-docx requests beautifulsoup4 lxml`

4. Change the name of '.env.example' to '.env' and add your Azure OpenAI deployment name, API key, and endpoint.

5. Run the script using the following command:

`python summarizer.py <input_path> <output_path> [--summary-level <summary_level>]`

Replace `<input_path>` with the path to the input file (text, PDF, Word) or URL, `<output_path>` with the path to the output summary file, and `<summary_level>` (optional) with one of the following options: "verbose", "concise", or "terse". If the `--summary-level` flag is not used, the default option is "concise".

Examples:

`python summarizer.py input.pdf text_summary.txt`

`python summarizer.py https://example.com/no-one-reads-long-articles-anymore.html url_summary.txt --summary-level terse`

These commands will generate a summary of the input document or URL and save it in the `*_summary.txt`* file.
