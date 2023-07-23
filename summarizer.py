# Import necessary libraries and modules
import argparse
import asyncio
import os
import re

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

# Define the default prompt
# default_prompt = """
# Analyze the content you are given and provide an executive summary of the content. The summary should include all information necessary for an executive to act upon.
# Ignore any extraneous information that is not relevant to an executive, including personal information, small talk, jokes, etcetera.
# As new content is given to you, revise the entire summary to include the new content. Be concise and complete. Focus on important details and key takeaways.
# If the content is a chat transcript, start the summary with a list of attendees as bullet points.
# If the content is a document, start the summary with the title of the document. If no title is present in the document, provide one based on the content summary.
# Do not include any extraneous information in the summary that is not part of the content. Only focus on the content itself.
# """
default_prompt = """
Summarize content concisely for executive action, including important details.
Update with new information.
For chats, list attendees; for documents, provide the title.
Ignore extraneous or non-relevant information.
Break the summary into paragraphs when appropriate for readability.
Retain the attendees list and/or title at the beginning of the summary."""


# Initialize the kernel and configure it
kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat-gpt", AzureChatCompletion(deployment, endpoint, api_key)
)

# Configure the prompt settings
prompt_config = sk.PromptTemplateConfig.from_completion_parameters(
    max_tokens=2000, temperature=0.7, top_p=0.8
)

prompt_template = sk.ChatPromptTemplate(
    "{{$user_input}}", kernel.prompt_template_engine, prompt_config
)

prompt_template.add_system_message(default_prompt)

# Set up the semantic function for summarization
function_config = sk.SemanticFunctionConfig(prompt_config, prompt_template)
document_summary_function = kernel.register_semantic_function(
    skill_name="Summarizer", function_name="Summarizer", function_config=function_config)


async def process_text(input_text):

    context_vars = sk.ContextVariables()
    context_vars["user_input"] = input_text

    summary = await kernel.run_async(document_summary_function, input_vars=context_vars)
    return summary


async def summarize_document(input_path, output_path):
    chunk_size = 3000
    previous_summary = None

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(input_path, "r") as f:
        with open(output_path, "w") as out_f:
            while True:
                print("Summarizing...")
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                if previous_summary:
                    # Compress the existing summary
                    # compressed_summary = await process_text(previous_summary, output_path, None, prompt)

                    # Combine the compressed summary with the input chunk
                    # input_text = str(compressed_summary) + "\n\n" + chunk
                    input_text = str(previous_summary) + "\n\n" + chunk
                else:
                    input_text = chunk

                # Perform summarization using the combined input
                summary = await process_text(input_text)

                if summary:
                    previous_summary = summary
                    print("Summary iteration: \n", summary)
                else:
                    print("No summary generated for the current chunk.")

            # Write the final summary to the output file
            if previous_summary:
                out_f.write(str(previous_summary))

    print("summary complete!")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Document Summarizer")
parser.add_argument("input_path", help="Path to the input text file")
parser.add_argument("output_path", help="Path to the output summary file")
parser.add_argument("-p", "--prompt", help="An alternative prompt (optional)")

args = parser.parse_args()

if __name__ == "__main__":
    asyncio.run(summarize_document(
        args.input_path, args.output_path))
