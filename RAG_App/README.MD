# Documentation

## Objective

The objective of this script is to build a question-answering system using movie review data. The system leverages embeddings and a sequence-to-sequence model to generate responses to user queries based on the content of movie reviews. Specifically, the script performs the following tasks:

1. Loads a dataset of movie reviews.
2. Computes embeddings for each review using a sentence transformer model.
3. Builds and saves a Faiss index for efficient similarity search.
4. Uses a sequence-to-sequence model to generate answers to user queries, utilizing the most relevant review context.

## Data Description

The dataset used in this script is a subset of the Rotten Tomatoes movie review dataset. The script loads the first 1000 samples from the training split of the dataset. Each sample consists of:

- **Text**: The movie review text, which serves as the primary document content.

## Methodology

### 1. Data Loading

The dataset is loaded using the `datasets` library.

### 2. Model Initialization

#### Sentence Transformer Model

The `paraphrase-MiniLM-L3-v2` model from the `sentence-transformers` library is used to encode the movie reviews into embeddings.

#### Sequence-to-Sequence Model

The `sshleifer/distilbart-cnn-6-6` model from the `transformers` library is used for generating answers based on the context and query.

### 3. Document Embeddings

Embeddings for each document are computed using the sentence transformer model.

### 4. Faiss Index Creation

A Faiss index is created and populated with the document embeddings for efficient similarity search.

### 5. Answer Generation

When a query is provided, the script:

1. Encodes the query into an embedding.
2. Searches for the most similar document using Faiss.
3. Generates an answer using the sequence-to-sequence model based on the query and the retrieved document context.

## Model Building

### Sentence Transformer Model

- **Model Used**: `paraphrase-MiniLM-L3-v2`
- **Purpose**: To encode text documents into dense vector embeddings.

### Sequence-to-Sequence Model

- **Model Used**: `sshleifer/distilbart-cnn-6-6`
- **Purpose**: To generate answers based on the query and context.

## Model Evaluation

Evaluation is done implicitly in the script through the output generated for sample queries. The effectiveness of the system is determined by:

- **Relevance of Context**: Ensuring that the context retrieved by the Faiss index is relevant to the query.
- **Quality of Generated Answer**: Evaluating the coherence and informativeness of the generated response.

## Benefits

- **Efficient Search**: Faiss provides an efficient mechanism to search for relevant documents based on query embeddings.
- **Contextual Answer Generation**: The sequence-to-sequence model generates answers using the most relevant context, enhancing the quality of responses.

## Conclusion

The script demonstrates a functional approach to building a question-answering system using embeddings and sequence-to-sequence generation. By leveraging Faiss for similarity search and a transformer-based model for generation, it provides a robust mechanism to address user queries based on a dataset of movie reviews. Future improvements could involve fine-tuning models or expanding the dataset for more comprehensive answers.
