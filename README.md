# Multi-Model Chat System with AI Judgment & Internet Search

## Overview

This project is an AI-powered multi-model chat system that runs multiple top-tier language models asynchronously and uses GPT-4.0 as a judge to determine the best response. It integrates real-time internet search capabilities to enhance response accuracy and relevance.

## Features

- **Multi-Model Execution:** Supports multiple LLMs, including OpenAI's GPT-4.0, GPT-4.0 Turbo, GPT-3.5, Olama, and Mistral.
- **Asynchronous Processing:** Runs multiple models in parallel, reducing response time and improving comparison accuracy.
- **AI Response Evaluation:** Uses GPT-4.0 as a judge to rank and determine the most suitable response.
- **Internet Search Integration:** Utilizes DuckDuckGo for real-time web search, ensuring responses are backed by the latest available information.
- **Interactive UI:** Built with Streamlit for an intuitive and user-friendly chat experience.
- **AI Orchestration:** Uses LangChain to manage AI model interactions and retrieval workflows.

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python, LangChain
- **AI Models:** OpenAI GPT-4.0, GPT-4.0 Turbo, GPT-3.5, Olama, Mistral
- **Search Engine:** DuckDuckGo API
- **Processing:** Asynchronous execution for parallel model response generation

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip
- OpenAI API Key (for GPT models)

### Setup Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/e-d-i-n-i/multi-model-chat.git
   cd multi-model-chat
   ```
2. Create a virtual environment and activate it:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Set up environment variables:
   ```sh
   export OPENAI_API_KEY="your_openai_api_key"
   export DUCKDUCKGO_API_KEY="your_duckduckgo_api_key"
   ```
5. Run the application:
   ```sh
   streamlit run app.py
   ```

## Usage

1. Open the Streamlit web interface.
2. Enter a query, and the system will generate responses from multiple AI models.
3. The best response, as judged by GPT-4.0, will be displayed prominently.
4. Click on other model responses to compare different answers.
5. Enable or disable internet search to enhance AI responses.

## License

This project is licensed under the MIT License.
