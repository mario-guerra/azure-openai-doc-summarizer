# Doc Summarizer with Azure OpenAI

Doc Summarizer is a Python script that uses Azure OpenAI models to generate rich summaries of large documents, including text files, PDFs, Word documents, and websites.

The script processes the input document in chunks based on the selected summary level (verbose, concise, terse, barney, or transcribe), iterating on the summary using a sliding content window for the context.

The sliding content window consists of the most recent paragraphs from the previous summary, plus the current input text chunk. This approach allows the script to iteratively summarize large text files without overflowing the model's token limits, while still retaining enough context between summarization steps to produce a cohesive summary of the entire document.

Read more:
[The Sliding Content Window: Document Summarization with Azure OpenAI](https://marioguerra.xyz/ai-document-summarization-with-sliding-content-window/)

## Dependencies

- Python 3.6 or later
- azure-identity
- openai
- requests
- PyPDF2
- python-docx
- beautifulsoup4
- lxml

You can install the required libraries using pip:

```sh
pip install azure-identity openai requests PyPDF2 python-docx beautifulsoup4 lxml
```

## Requirements

An Azure OpenAI subscription is required to run this script, along with a deployment for one of the following models:
- gpt-3.5-turbo
- gpt-4
- gpt-4-32k

The difference in models are tradeoffs between performance and speed. The gpt-4 model produces the best summary results, but gpt-35-turbo is faster. In testing with technical content, gpt-4 was better than the other two models at analyzing and retaining technical aspects of the content. The other two models produce decent summaries, but potentially important details are lost.

## Overview

The script reads the input document in chunks and uses the Azure OpenAI API to generate summaries. Token rate limit errors and timeouts are handled with retry logic. The final summary is written to the specified output file.

The input document can be a plain text file, a PDF document, a Word document, or a URL. The script automatically detects the input file type and extracts the text accordingly.

## Usage

1. Clone the repository to your local machine.

2. Open a terminal and navigate to the folder containing the script file.

3. Install the dependencies outlined above using pip:

```sh
pip install azure-identity openai requests PyPDF2 python-docx beautifulsoup4 lxml
```

4. Create a `.env` file in the project directory and add your Azure OpenAI deployment name, API key, and endpoint:

```
ENDPOINT_URL=https://your_endpoint
DEPLOYMENT_NAME=your_deployment_name
```

5. Run the script using the following command:

```sh
python summarizer.py <input_path> <output_path> [--summary-level <summary_level>] [--prompt <custom_prompt>]
```

Replace `<input_path>` with the path to the input file (text, PDF, Word) or URL, `<output_path>` with the path to the output summary file, `<summary_level>` (optional) with one of the following options: "verbose", "concise", "terse", "barney", or "transcribe". If the `--summary-level` flag is not used, the default option is "verbose". Optionally, you can add a custom prompt using the `--prompt` flag.

Examples:

```sh
python summarizer.py input.pdf text_summary.txt
```

```sh
python summarizer.py https://example.com/no-one-reads-long-articles-anymore.html url_summary.txt --summary-level terse
```

These commands will generate a summary of the input document or URL and save it in the specified output file.