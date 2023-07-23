# Import necessary libraries and modules
import argparse
import asyncio
import os
import re

import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

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
    chunk_size = 20000
    previous_summary = None
    total_characters = 0  # Initialize the total characters count
    summarized_characters = 0  # Initialize the summarized characters count

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(input_path, "r") as f:
        total_characters = len(f.read())  # Get the total characters count
        f.seek(0)  # Reset the file pointer to the beginning

        with open(output_path, "w") as out_f:
            while True:
                print("Summarizing...")
                chunk = f.read(chunk_size)

                if not chunk:
                    break

                # Update the summarized characters count
                summarized_characters += len(chunk)
                # Display current progress
                print(f"Progress: {summarized_characters}/{total_characters}")

                if previous_summary:
                    input_text = str(previous_summary) + "\n\n" + chunk
                else:
                    input_text = chunk

                summary = await process_text(input_text)

                if summary:
                    previous_summary = summary
                    print("Summary iteration: \n", summary)
                else:
                    print("No summary generated for the current chunk.")

            if previous_summary:
                out_f.write(str(previous_summary))

    print("summary complete!")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Document Summarizer")
parser.add_argument("input_path", help="Path to the input text file")
parser.add_argument("output_path", help="Path to the output summary file")

args = parser.parse_args()

if __name__ == "__main__":
    asyncio.run(summarize_document(
        args.input_path, args.output_path))
