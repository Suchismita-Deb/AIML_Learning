## RAG Questions.

<div class="quiz-box">
<strong>What is RAG and what problem does it solve?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>Explain all the components of the RAG pipeline end to end.</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>When should you use RAG over fine-tuning?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>RAG vs a longer context window — which wins, and when?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What are the failure modes of a basic RAG system?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>Explain the difference between extractive and generative QA.</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>When is using RAG the wrong decision?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

## Chunking.

<div class="quiz-box">
<b>What is Chunking?</b>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
Chunking is breaking down large pieces of text into a smaller segments before embedding them and storing them into vector DB.<br>
The main reason to do this chunking is to embed the pieces of content with minimal noise and maintaining the semantic relevance.<br>
In conversational agents good chunking gets irrelevant context for users query and also makes sure it fits the LLM token limits.<br>

Consider the length of the embedded data when embedding short content  - The vectors focuses on a very specific meanings and excel at the precise matching but they do miss the broader context. When embedding long content - The vector captures the theme and the relationship and providing a comprehensive representation but it may dilute significance of details.

Chunking consideration.

To identify what level of chunking to use if person to identify what type of data we need to work - Long form or research level of content or short form or tweet level of content.

Short query works better with sentence level embeddings and long and more complex queries work with paragraph level embeddings.

Embedding models - Identify the embedding models we are going to use different model performs with different chunks. Example sentence transformer model works better with individual sentences whereas the text Embedding Ada 002 works better with 256 or 512 token chunks.

Consider the quesry expectation - short specific question or more complex?
</details>
</div>

<div class="quiz-box">
<strong>How do you pick the right chunk size for different use cases?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
To determine the right chunk size when it considered few of the cases - 
Preprocess the data remove all the noise and the relevant content from the data. 
Range of the chunk sizes - To play with the different embeddings model and types. 
Evaluate the performance - for evaluating metrics like relevance accuracy and retrieval speeds. First pick smaller sizes like 128 to 2C56 tokens to accommodate more granular informations.
</details>
</div>

<div class="quiz-box">
<strong>What are the chunking strategies, and when should you use them?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
Fixed Size chunking - It's the most straightforward approach and each chunk equals a token.
We decide how many tokens each chunk should contain and then split the text accordingly. We can add overlaps to pros of the context of the content. This method is simple to implement and works well when the content is relatively uniform in structure. However, it may split sentences or paragraphs in unnatural places, which can lead to loss of context and meaning.

```python
from langchain.text_splitter import CharacterTextSplitter

# Sample document
document = """This is a long HR policy document. It contains rules about dress code,
leave regulations, and code repository access. Employees must follow these guidelines strictly..."""

# Initialize splitter
text_splitter = CharacterTextSplitter(
separator="\n",       # split on newline
chunk_size=100,       # max characters per chunk
chunk_overlap=20,     # overlap between chunks
length_function=len   # function to measure length
)

# Split into chunks
chunks = text_splitter.split_text(document)
# Print chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
    
# The output.
# Chunk 1: This is a long HR policy document. It contains rules about dress code.
# Chunk 2: leave regulations, and code repository access. Employees must follow these guidelines strictly...

```

Content-aware chunking - It is also known as sentence splitting. Many of the embedding models are optimized for sentence level content with NLP libraries.

NLTK-based text splitting with content-aware chunking is a step up from fixed-size splitting. Instead of blindly cutting text every N characters, you use linguistic boundaries (sentences, paragraphs) to keep chunks meaningful. This is especially useful for your 100K-document chatbot pipeline, since embeddings work better when chunks preserve semantic context.
```python
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

def nltk_content_chunking(text, max_chunk_size=500):
    """
    Splits text into content-aware chunks using NLTK sentence tokenizer.
    
    Parameters:
        text (str): Input document string.
        max_chunk_size (int): Maximum characters per chunk.
    
    Returns:
        List[str]: List of content-aware text chunks.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


# Example usage
document = """This is a long HR policy document. It contains rules about dress code.
Employees must follow these guidelines strictly. Leave regulations are also included.
Code repository access is restricted to authorized users."""

chunks = nltk_content_chunking(document, max_chunk_size=80)

for i, c in enumerate(chunks):
    print(f"Chunk {i+1}: {c}") 
    
# The output.
/*
Chunk 1: This is a long HR policy document.
Chunk 2: It contains rules about dress code.
Chunk 3: Employees must follow these guidelines strictly.
Chunk 4: Leave regulations are also included.
Chunk 5: Code repository access is restricted to authorized users.
*/
```

Recursive chunking - It's an intelligent division of the text using a set of separators to reach the desired chunk size. Recursive chunking is a smarter way of splitting documents into chunks for LLMs. Instead of cutting text blindly at fixed sizes, it tries to preserve semantic boundaries (paragraphs, sentences) first, and only falls back to smaller splits if needed. This makes it ideal for your 100K-document chatbot pipeline, since embeddings work better when chunks are meaningful.
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Sample document
document = """This is a long HR policy document. It contains rules about dress code.
Employees must follow these guidelines strictly. Leave regulations are also included.
Code repository access is restricted to authorized users."""

# Initialize recursive splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,       # max characters per chunk
    chunk_overlap=20,     # overlap between chunks
    separators=["\n\n", "\n", ".", " "]  # hierarchy: paragraph → sentence → word
)

# Split into chunks
chunks = text_splitter.split_text(document)

# Print chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")

# The output.
Chunk 1: This is a long HR policy document. It contains rules about dress code.
Chunk 2: Employees must follow these guidelines strictly.
Chunk 3: Leave regulations are also included.
Chunk 4: Code repository access is restricted to authorized users.

```
</details>
</div>

<div class="quiz-box">
<strong>How do you handle tables, code, and images during chunking?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you pick an embedding model for production?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>When should you train or fine-tune embeddings?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What is the parent-document retrieval pattern?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

## VectorDb and Retrievals.

<div class="quiz-box">
<strong>Compare pgvector, Pinecone, Weaviate, Qdrant, and Chroma. When should you use each one?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
pgvector when you want Postgres-native simplicity, Pinecone for managed enterprise scale, Weaviate for open-source flexibility with built-in ML modules, Qdrant for high-performance metadata filtering, and Chroma for lightweight developer-friendly RAG prototypes.
</details>
</div>

<div class="quiz-box">
<strong>What is HNSW and why is it the default index for vector databases?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>Dense versus sparse retrieval: when should you use which one, and when do you need both?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What is Hybrid Search and how would you implement it?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How does re-ranking work and when is it worth the latency?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you implement metadata filtering at scale?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you handle multi-tenant data isolation in a vector database?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

## Evaluation.

<div class="quiz-box">
<strong>How do you evaluate a RAG system end to end?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What metrics measure generation quality (faithfulness, relevance)?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you build a golden dataset for RAG?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What is LLM as judge for RAG and what are its limits?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you detect retrieval drift in production?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you regression test RAG in CI/CD?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

## Production and Advanced.

<div class="quiz-box">
<strong>How do you reduce hallucinations in a RAG system?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you force the LLM to cite sources reliably?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What is HyDE (Hypothetical Document Embedding)?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>What is query rewriting and when do you need it?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you handle multi-hop or multi-document questions?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>How do you manage cost and latency in production RAG?</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
</details>
</div>

<div class="quiz-box">
<strong>Design a production RAG system for a 10M document corpus.</strong>
<details class="quiz-toggle">
<summary>Reveal Answer</summary>
Companies want validated enterprise level system. The rag must be production ready and the decision and safety layers are critical.<br>

Mention that word RAG as Retrieval knowledge from internal sources. Augment LLM reasoning. Producing validated and context-aware answers.

Companies can use the RAG stack for information processing. RAG is the retrieval of the internal knowledge with validated answers.<br>

The ask is to design a RAG pipeline then follow 3 parts - 
<b>Knowledge Layer</b> - It is the knowledge of the internal document. It can be policy document, domain specific document, compliance document. <br>
Vector DB for semantic search - Perform the chunking on the knowledge layer - split the document and save in vector format in the embedding documents and it will stored in vector db to perform the semantic search. <br> Metadata driven filtering It nevers stores the chunks itslef it is stored with the metadata.<br>

<b>Retrieval Layer</b> - It is the layer we do the semantic search and retrieve the relevant documents from the vector db. It will apply the security filters and based on that we will retrieve the top k results. It will act as inputs to the LLM.<br>
Domain and access filtering and it is the vaidated retrieveal of data and we will only retrieve domain related knowledge.
Validation Layer - Cite the source,  confidence scre and validate to show the output.


Example - Incident resolution - When an incident comes the agents retrieve runbooks and operational docs and validate the confidence scoring and then propose the resolution.
Mention validate retrieval and not retrieval.  
Mention internal knowledge with security controls.  
Citation and confidence scoring.
human approvals for high-risk workflow.

Follow Safety-governance-validation-approval state.
Sample - The RAG pipeline retrieves the internal knowledge using the vector db. It validates the information with the citation and confidence score. It produces the safe context aware answers which are suitable for the LLM to use. Example incident resoution and support.
</details>
</div>



