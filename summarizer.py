# Import necessary libraries and modules
import argparse
import asyncio
import os
import re
from tiktoken import Tokenizer as TikTokenizer
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

model_max_tokens = {
    "gpt-4": 8192,
    "gpt-4-32k": 32768,
    "gpt-35-turbo": 4096,
    "gpt-35-turbo-16k": 16384,
}

# Conservatively estimating 5 characters per token
model_max_chars {
    "gpt-4": model_max_tokens["gpt-4"]/5,
    "gpt-4-32k": model_max_tokens["gpt-4-32k"]/5,
    "gpt-35-turbo": model_max_tokens["gpt-35-turbo"]/5,
    "gpt-35-turbo-16k": model_max_tokens["gpt-35-turbo-16k"]/5,
}

default_prompt = """
Summarize content concisely for executive action, including important details.
Update with new information.
For chats, list attendees; for documents, provide the title.
Ignore extraneous or non-relevant information.
Break the summary into paragraphs when appropriate for readability."""
#If Retain the attendees list and/or title at the beginning of the summary."""

# Initialize the kernel and configure it
kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service(
    "chat-gpt", AzureChatCompletion(deployment, endpoint, api_key)
)

tik_tokenizer = TikTokenizer()

def estimate_token_count(text):
    tokens = list(tik_tokenizer.tokenize(text))
    return len(tokens)

max_tokens = model_max_tokens[deployment] - estimate_token_count(default_prompt)

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

def write_paragraphs(out_f, paragraphs):
    for p in paragraphs:
        out_f.write(p + "\n\n")

def extract_summary_paragraphs(summary_text):
    paragraphs = str(summary_text).split('\n\n')
    return [p.strip() for p in paragraphs]

async def summarize_document(input_path, output_path):
    chunk_size = 10000
    max_context_paragraphs = 5
    max_tokens = 3000  # Update this as needed(change based on model's token limit minus the tokens required for the prompt)
    previous_summary_paragraphs = []

    if os.path.exists(output_path):
        os.remove(output_path)

    with open(input_path, "r") as f:
        total_chars = os.stat(input_path).st_size

        with open(output_path, "a") as out_f:
            processed_chars = 0
            while True:
                print("Summarizing...")
                chunk = f.read(chunk_size)
                processed_chars += len(chunk)

                if not chunk:
                    break

                input_text = "\n\n".join(previous_summary_paragraphs[-max_context_paragraphs:]) + "\n\n" + chunk
                
                # Ensure input_text token count does not exceed the model's token limit
                input_tokens = estimate_token_count(input_text)
                while input_tokens > max_tokens:
                    removed_paragraph = previous_summary_paragraphs.pop(0)
                    out_f.write(removed_paragraph + "\n\n")
                    input_text = "\n\n".join(previous_summary_paragraphs[-max_context_paragraphs:]) + "\n\n" + chunk
                    input_tokens = estimate_token_count(input_text)

                summary_ctx = await process_text(input_text)

                # Convert the SKContext type to a string
                summary = str(summary_ctx)

                if summary:
                    summary_paragraphs = extract_summary_paragraphs(summary)
                    overflow_paragraphs = len(previous_summary_paragraphs) + len(summary_paragraphs) - max_context_paragraphs
                    if overflow_paragraphs > 0:
                        write_paragraphs(out_f, previous_summary_paragraphs[:overflow_paragraphs])
                        previous_summary_paragraphs = previous_summary_paragraphs[overflow_paragraphs:]
                    previous_summary_paragraphs.extend(summary_paragraphs)
                    print("Summary iteration: \n", summary)
                else:
                    print("No summary generated for the current chunk.")

                # Print the progress counter
                progress = (processed_chars / total_chars) * 100
                print(f"Progress: {processed_chars}/{total_chars} ({progress:.2f}%)")

            # Write the final summary paragraphs to the output file
            write_paragraphs(out_f, previous_summary_paragraphs)

    print("Summary complete!")

# Parse command line arguments
parser = argparse.ArgumentParser(description="Document Summarizer")
parser.add_argument("input_path", help="Path to the input text file")
parser.add_argument("output_path", help="Path to the output summary file")
parser.add_argument("-p", "--prompt", help="An alternative prompt (optional)")

args = parser.parse_args()

if __name__ == "__main__":
    asyncio.run(summarize_document(
        args.input_path, args.output_path))
