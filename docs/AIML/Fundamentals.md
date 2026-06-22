# Fundamentals

AI agents are built from multiple components that work together to understand tasks, reason, and act.

## Core Parts of AI Agents

-LLMs.  
-RAGs.  
-VectorDB.
-Langchain/Langgraph.  
-MCP Server.  
-Prompts.    
-Tokens.  
-Embeddings.   
-Context window.
-Agents.   
-Claude/Gemini.

**LLM** - OpenAI LLM is GPT, Anthropic LLM is Claude, Google LLM is Gemini. They are all transformer model trained on large sets of data.

The training data is a mixture of licensed data, data created by human trainers, and publicly available data. The training process involves trillions of data running computations on thousands of GPUs over weeks or months, which is why it is expensive.

LLM to onverse need a short term memory to remember the conversation This is called the short term memory. The Context Window (short term memory) is the amount of text that the model can process at a time. It includes both the prompt and the generated response. For example, if a model has a context window of 4096 tokens, it can process a total of 4096 tokens in the prompt and response combined.

Context window is  measured with token whih is 3/4th of the word. So 4096 tokens is around 3000 words.   
If the conversation exceeds the context window, the model may lose track of earlier parts of the conversation, which can lead to less coherent responses.

Example xAi Grok 4 has 256k tokens and Anthropic Opus 4 has 200k tokens and Google Gemini 2.5Pro has 1M tokens. The larger the context window, the more information the model can consider when generating responses, which can lead to more accurate and contextually relevant answers.

The amount of data a model store in the context window varies. The model like nano, mini, flash has 2k-4k tokens meaning 1500-3000 words. The big model like GPT4.1, Gemini 2.5Pro has 1M tokens meaning 750K words and 50k lines of code.

In tech env 500Gb of the repo documents are not possible to load in context window. The largest model can store around 50 files of business documents. To make the model learn about the entire thing we introduced Embeddings.

**Embeddings** makes the meaning into numbers.  
The term employee vacation policy and staff time off are different words and the meaning is same. It is considered as semantics similarity. The embedding model converts the text into a vector of numbers like 1536 numbers that represents the meaning. The same concept ends up with same number. Example vacation and holiday will have the same number.

Similar concepts ends up with similar vectors. So the model can understand that employee vacation policy and staff time off are similar concepts even though they are different words. It will find the relevant document based on what someone needs and not words.  

Langchain/Langgraph is a framework that helps to build applications with LLMs. It provides tools and abstractions to manage prompts, handle conversations, and integrate with external data sources. 

The chatbot needs to answer company policy, product info, support issues. The chatbot store conversation history, complex multi-step interactions. The instinct to use the openAi sdk to chat but the missing piece is the storing chat messages, maintaining context and connecting to internal documents. 

In case the company switch from anthropic to google then its a big task. We need to write code to connect them. There is an abstraction layer available Langchain.

Langchain - An abstraction later that helps to build AI agents with minimal code.  

LLM uses static brain nd Agent uses tools, autonomy, memory.
```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain.memory import MemorySaver
from langchain.chains import ConversationalRetrievalChain

# Choose your LLM (uncomment the one you want to use)
# llm = ChatOpenAI(model="gpt-3.5-turbo")
llm = ChatAnthropic(model="claude-3-sonnet")

# Initialize memory and embeddings
memory = MemorySaver() // It will store and retrieve chat history. There is no need to build own db schema for session management.
embedding = OpenAIEmbeddings()

# Create Chroma database collection
db = Chroma(collection_name="techcorp_docs", embedding_function=embedding)
// Vector db works through standard interface. So if we want to switch to other vector db like pinecone, we can do it with minimal code changes.
// Text ebedding to conert similar things to vector.

# Build the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory
)

// Tool integration allows agents to access internal systems. To query customer database we need to make a tool that agent can call.

// Lang chin is doing all for me starting from API to call the LLM memory, creating the vector DB and embedding pipelines for semantic searches, tool routing and state management.

# Run a query
response = qa_chain.run("What is TechCorp’s customer data policy?")
print(response)
// We can increase the performance by adding more tools like custom database access web search and local file systems.
```


### Code to show the use of API key and OpenAi client to send prompt and os to read the API key from the env variable.

```jupyter
import os
from openai import OpenAI
// OpenAI is the company that created Chatgpt and build the AI models. The Openai Python library is the gateway to this AI models.

// The OS library is important to access the environment variables starting the key and all other details.

# Load API key from environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
base_url = os.getenv("OPENAI_API_BASE")
client = OpenAI(api_key=api_key, base_url=base_url)

# The base URL tells the client where to send the request if the address to the open AI server. The SDK points to OpenAI public endpoint by default. 

# Define your prompt
prompt = "Write a short poem about sunrise over the ocean."

# Send request to Chat Completions endpoint
response = client.chat.completions.create(
    model="gpt-4o-mini",   # choose model
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.7,       # creativity level
    max_completion_tokens=150  # response length
)
// client.chat.completions.create is OpenAi's conversational API. 


// There are 3 roles in conversation - `system` - Instructions the way AI should behave - `user` - the question and the `assistant` - The AI's response.

# Print the generated text. choices is an array of the responses.
print(response.choices[0].message.content)

# Print token usage details. Tokens are the pieces of word that AI use in the process every request. Rough estimate 1 token = 4 character.
There are three token types prompt token the question you asked completion token the AI answers and total token is the summation of both.
print("Prompt tokens:", response.usage.prompt_tokens)
print("Completion tokens:", response.usage.completion_tokens)
print("Total tokens:", response.usage.total_tokens)

```

It will complete the text or generate the idea or describe. The response is non-deterministic and there ways to manage the amount of randomness.

Temperature is used to create the amount of randomness.

Langchain - Multimodel A/B testing or balance cost in all models. It will be used to evaluate all models.
```python
import os
from langchain_openai import ChatOpenAI

def main():
    prompt = "Explain artificial intelligence in exactly 5 words"
    # Initialize OpenAI GPT-4.1-mini.
    openai_llm = ChatOpenAI(
        model="openai/gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7
    )
    # Initialize Google Gemini 2.5-flash.
    gemini_llm = ChatOpenAI(
        model="google/gemini-2.5-flash",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7
    )
    # Run prompt against both models
    openai_response = openai_llm.invoke(prompt)
    gemini_response = gemini_llm.invoke(prompt)
    print("\nRESULTS COMPARISON")
    print(f"OpenAI GPT-4.1-mini: {openai_response.content}")
    print(f"Google Gemini 2.5-flash: {gemini_response.content}")

if __name__ == "__main__":
    main()
```


Output parsers - Tools that transform unstructured AI text into a structured Python data. It will convert to list, dict.

**Chain Composition** -  Connecting Langchain  components with the | operator to create data pipelines like Unix pipes for AI.
The data flow from left to right.
`prompt | llm | parser | db_save | email_notify`


```python
"""
Task 4: Output Parsers – From Text to Structured Data. Transform AI responses into structured formats that applications can use.
Learning Goal: Extract structured data from unstructured AI responses.
"""

import os
import json
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate

def main():
    # Initialize LLM.
    llm = ChatOpenAI(
        model="openai/gpt-4.1-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0.7
    )
    
    # Example 1: List Parser – Simple structured data.
    list_parser = CommaSeparatedListOutputParser()

    list_prompt = PromptTemplate(
        template="List 3 benefits of {technology} (comma-separated):",
        input_variables=["technology"]
    )

    # Build chain with list parser
    list_chain = list_prompt | llm | list_parser

    # Test the list chain
    list_result = list_chain.invoke({"technology": "cloud computing"})
    print("Parsed List:", list_result)

    # Example 2: JSON Output – Complex structured data
    
    json_prompt = PromptTemplate(
        template="""Analyze {technology} and respond with JSON containing:
        - benefits: array of 2 benefits
        - complexity: low/medium/high
        - use_case: one main use case

        Technology: {technology}

        Respond ONLY with valid JSON:""",
        input_variables=["technology"]
    )

    # Use a string parser first, then load JSON
    json_chain = json_prompt | llm | StrOutputParser()

    json_result = json_chain.invoke({"technology": "machine learning"})
    try:
        parsed_json = json.loads(json_result)
        print("Parsed JSON:", parsed_json)
    except json.JSONDecodeError:
        print("️ Failed to parse JSON:", json_result)

if __name__ == "__main__":
    main()
    

"""
The output will be.

List Output Parser.
Parsed List: ['Scalability', 'Cost efficiency', 'Flexibility']

JSON Output Parser.
Parsed JSON: {
  "benefits": ["Automates tasks", "Improves decision-making"],
  "complexity": "high",
  "use_case": "Predictive analytics"
}
"""
```

Prompt engineering will help the agent to give far more accurate reply. 

There are different types of prompting like zero shot, few shot, chain of thought, self consistency, tree of thought, active prompting and prompt tuning. 

Zero shot prompting is when you give the model a task without any examples.   
Few shot prompting is when you provide a few examples to guide the model.  
Chain of thought prompting encourages the model to reason step-by-step.   
Self-consistency prompting involves generating multiple answers and selecting the most consistent one.   
Tree of thought prompting explores multiple reasoning paths.   
Active prompting involves interacting with the model to refine its responses.   
Prompt tuning is a method of fine-tuning prompts for specific tasks.


```python
def run_prompt(prompt, label):
    print(f"\n--- {label} ---")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=200,
        temperature=0.7
    )
    print(response.choices[0].message.content)

def main():
    # 1. Zero-shot prompting
    run_prompt("Translate 'Hello, how are you?' into French.", "Zero-shot")

    # 2. Few-shot prompting
    few_shot_prompt = """Translate the following into French:
    Example: 'Good morning' -> 'Bonjour'
    Example: 'Thank you' -> 'Merci'
    Now translate: 'Hello, how are you?'"""
    run_prompt(few_shot_prompt, "Few-shot")

    # 3. Chain-of-thought prompting
    cot_prompt = """Solve step by step:
    Q: If a train travels 60 km in 1 hour, how far will it travel in 4 hours?
    A:"""
    run_prompt(cot_prompt, "Chain-of-thought")

    # 4. Self-consistency prompting
    # Generate multiple answers and compare consistency
    for i in range(3):
        run_prompt("What is 12 * 13?", f"Self-consistency run {i+1}")

    # 5. Tree-of-thought prompting
    tree_prompt = """Consider multiple reasoning paths:
    Problem: Should I invest in stock A or stock B?
    Path 1: Analyze risk.
    Path 2: Analyze growth potential.
    Path 3: Analyze market trends.
    Provide a conclusion after exploring all paths."""
    run_prompt(tree_prompt, "Tree-of-thought")

    # 6. Active prompting (interactive refinement)
    run_prompt("Write a short poem about the ocean.", "Active prompting - initial")
    run_prompt("Make the poem rhyme and limit to 4 lines.", "Active prompting - refinement")

    # 7. Prompt tuning (specialized instruction style)
    tuned_prompt = """You are a financial analyst.
    Task: Summarize the benefits of mutual funds in 3 bullet points."""
    run_prompt(tuned_prompt, "Prompt tuning")

if __name__ == "__main__":
    main()
```
**Vector Db** - In company say there is 

25:254

LLM to ask question about the tech internal documents we need the ability to pass data to the LLM. This is done by RAG (Retrieval Augmented Generation) which is a technique that combines the power of LLMs with external knowledge sources to improve the quality and relevance of generated responses. RAG allows LLMs to access and retrieve information from external sources, such as databases, documents, or APIs, to provide more accurate and contextually relevant answers.

