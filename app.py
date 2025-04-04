import os
import streamlit as st
from langchain_groq import ChatGroq
from langsmith import trace
from langchain.schema import HumanMessage, AIMessage
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun 
from langchain_openai import ChatOpenAI
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

# Load environment variables
load_dotenv()

# Set LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "RAG Chat Bot Multi Models"

# Initialize models (added GPT-3.5-turbo)
models = [
    {"name": "Llama-3.3-70b-Versatile", "model": "llama-3.3-70b-versatile"},
    {"name": "Deepseek-R1-Distill-Llama-70b", "model": "deepseek-r1-distill-llama-70b"},
    {"name": "Llama-3.1-8b-Instant", "model": "llama-3.1-8b-instant"},
    {"name": "Mixtral-8x7b-32768", "model": "mixtral-8x7b-32768"},
    {"name": "GPT-3.5-Turbo", "model": "gpt-3.5-turbo"},
]

# Initialize chat model clients
groq_client = ChatGroq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
openai_judge_client = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini")

# Initialize DuckDuckGo Search Tool
search_tool = DuckDuckGoSearchRun(name="Search")  # Use DuckDuckGo instead of Tavily

# Function to optimize user query using GPT-3.5-turbo
def user_query_optimizer(user_query: str) -> str:
    """
    Optimizes the user query by removing unnecessary words and making it a better prompt.
    """
    optimization_prompt = f"""
    You are an AI assistant tasked with optimizing user queries. 
    Please refine the following query by removing unnecessary words and making it more concise and clear:
    
    Query: {user_query}
    
    Optimized Query:
    """
    
    # Query the GPT-3.5-turbo model for query optimization
    optimized_query_response = openai_client.invoke([HumanMessage(content=optimization_prompt)])
    optimized_query = optimized_query_response.content.strip()
    
    return optimized_query

# Function to determine if a query needs a search
def needs_search(user_query: str) -> int:
    """
    Determines if the user query requires a search.
    Returns 1 if a search is needed, 0 otherwise.
    """
    # List of keywords or phrases that typically require a search
    search_keywords = ["who", "what", "when", "where", "why", "how", "current", "latest", "recent", "today", "2023", "2024"]
    
    # Check if any keyword is in the query
    if any(keyword in user_query.lower() for keyword in search_keywords):
        return 1  # Needs search
    else:
        return 0  # Does not need search

def perform_search(user_query: str) -> str:
    """
    Optimizes the search query using the Llama-3.1-8b-Instant model and performs a search using DuckDuckGo.
    """
    # Step 1: Optimize the query using Llama-3.1-8b-Instant
    optimization_prompt = f"""
    You are an AI assistant tasked with optimizing search queries. 
    Please condense the following query into 3-7 keywords that are most relevant for a web search:
    
    Query: {user_query}
    
    Optimized Keywords:
    """
    
    # Query the Llama-3.1-8b-Instant model for query optimization
    optimized_query_response = groq_client.invoke([HumanMessage(content=optimization_prompt)])
    optimized_query = optimized_query_response.content.strip()

    # Step 2: Perform the search using DuckDuckGo with the optimized query
    search_results = search_tool.run(optimized_query)
    
    return search_results

# Function to query models synchronously with timeout
def query_model(model_info, user_query, timeout=15):
    """Query a specific AI model and return its response and reasoning, with a timeout."""
    try:
        with trace(name=f"Query {model_info['name']}"):
            # Synchronously query the model (no need to await)
            if model_info["model"] == "gpt-3.5-turbo":
                completion = openai_client.invoke([HumanMessage(content=user_query)])
            else:
                completion = groq_client.invoke([HumanMessage(content=user_query)])
            response = completion.content

            # Simulate reasoning process
            reasoning = f"{model_info['name']} analyzed the query and generated this response."

            return model_info["name"], response, reasoning
    except Exception as e:
        return model_info["name"], f"Error: {str(e)}", "An error occurred while querying the model."

# Function to evaluate responses using OpenAI's model
def evaluate_responses(user_query, model_responses):
    """Evaluate the responses using OpenAI's model to determine the best response."""
    # Prepare the prompt for evaluation
    prompt = f"""
    You are an AI judge tasked with evaluating the quality of responses to the following query:
    Query: {user_query}

    Responses:
    {chr(10).join([f"Model: {model}, Response: {response}" for model, response, _ in model_responses])}

    Please evaluate each response based on relevance, accuracy, and completeness. Assign a score out of 100 to each response.
    Return the results in the format:
    Model: <model_name>, Score: <score>
    """
    
    # Query OpenAI's model for evaluation
    evaluation = openai_judge_client.invoke([HumanMessage(content=prompt)])
    evaluation_text = evaluation.content

    # Parse the evaluation results
    scores = {}
    for line in evaluation_text.splitlines():
        if "Model:" in line and "Score:" in line:
            model = line.split("Model: ")[1].split(",")[0].strip()
            score = int(line.split("Score: ")[1].strip())
            scores[model] = score

    return scores

# Handle multiple model queries
def run_queries(user_query):
    """Run queries synchronously for all models with timeout handling."""
    model_responses = [query_model(m, user_query) for m in models]
    scores = evaluate_responses(user_query, model_responses)
    return model_responses, scores

# Initialize session state attributes
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "steps" not in st.session_state:
    st.session_state.steps = {}
if "best_score" not in st.session_state:
    st.session_state.best_score = 0  # Initialize best_score

# Streamlit UI
st.set_page_config(page_title="Multi-Model AI Chat with Search", page_icon="🤖")
st.title("🤖 Multi-Model AI Chat with Search")

# Add a toggle to enable/disable searching
search_enabled = st.toggle("Enable Search", value=True)

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "human":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "ai":
        if "best" in message:
            st.chat_message("assistant").write(f"🤖 **{message['model']} (Best, {st.session_state.get('best_score', 0)}%)**: {message['content']}")
        else:
            with st.expander(f"⚡ {message['model']} ({message['score']}%)"):
                st.write(message["content"])
                st.write(f"🧠 **Reasoning:** {message['reasoning']}")

# User input
user_query = st.chat_input(placeholder="Ask something...")

if user_query:
    # Optimize the user query
    optimized_query = user_query_optimizer(user_query)
    
    # Add human message to chat history
    st.session_state.chat_history.append({"role": "human", "content": optimized_query})

    # Check if the query requires a search and if search is enabled
    if search_enabled and needs_search(optimized_query):
        # Perform search using DuckDuckGo and append the results to the query
        search_result = perform_search(optimized_query)
        optimized_query = f"{optimized_query}\n\nSearch Result: {search_result}"

    # Run AI models and evaluate responses
    model_responses, scores = run_queries(optimized_query)

    # Find the best response
    best_model = max(scores, key=scores.get)
    st.session_state.best_score = scores[best_model]  # Store best_score in session state

    # Store responses in chat history
    for model, response, reasoning in model_responses:
        st.session_state.chat_history.append(
            {"role": "ai", "model": model, "content": response, "score": scores[model], "reasoning": reasoning}
        )

    # Store best response separately
    best_response = next(item for item in model_responses if item[0] == best_model)
    st.session_state.chat_history.append({"role": "ai", "model": best_model, "content": best_response[1], "best": True})

    # Rerun to update the chat history display
    st.rerun()