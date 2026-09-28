import os

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# Load environment variables from .env
load_dotenv()

# Get Hugging Face token
hf_token = os.getenv("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN was not found. Check your .env file.")


# Create Hugging Face client
client = InferenceClient(
    api_key=hf_token
)


def chat_with_ai(message, history, instructions):

    # Start with the system instruction
    messages = [
        {
            "role": "system",
            "content": instructions
        }
    ]

    # Add previous conversation
    for item in history:

        if item["role"] in ["user", "assistant"]:
            messages.append({
                "role": item["role"],
                "content": item["content"]
            })

    # Add the current user message
    messages.append({
        "role": "user",
        "content": message
    })

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )

        assistant_reply = response.choices[0].message.content

        return assistant_reply

    except Exception as e:
        return f"Something went wrong: {e}"


# Create the chatbot interface
demo = gr.ChatInterface(
    fn=chat_with_ai,
    additional_inputs=[
        gr.Textbox(
            label="Custom Instructions",
            value="You are a helpful AI assistant. Explain things clearly and simply for beginners."
        )
    ],
    title="🤖 My AI Chatbot",
    description="A beginner-friendly AI chatbot powered by Hugging Face.",
    textbox=gr.Textbox(
        placeholder="Ask me anything...",
        label="Your Message"
    )
)


if __name__ == "__main__":
    demo.launch()