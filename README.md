# Clementine-Assistant

Clementine-Assistant is a Q&A application designed to assist developers and data scientists in working with SageMaker. It was developed using the Retrieval Augmented Generation (RAG) technique, which enhances Language Model Models (LLMs) by integrating retrieval mechanisms.

This approach ensures that the content generated by LLMs is both contextually relevant and inherently accurate. RAG serves as a bridge, connecting LLMs to extensive knowledge sources, thereby eliminating the need to train or fine-tune a new LLM model. This significantly reduces the time and cost required to develop this tool.

# Implementation

To implement RAG, we utilized the following components:

1. **Text Loader and Splitter**: We employed Langchain's classes to read the Markdown (md) files and segment them into smaller chunks. The chunks should be:


  - Large enough to contain enough information to answer a question.
  - Small enough to fit into the LLM prompt: **google/flan-t5-large** input tokens are limited to 512 tokens
  - Small enough to fit into the embeddings model: **all-MiniLM-L6-v2** input tokens limited to 512 tokens (roughly 2000 characters  if we assume 1 token \~ 4 characters).

      Given the above, we are going to use a chunk size of **1000** with overlap of **300** and the retrieval returning only the $2$ most similar chunks.


2. **Vector Store**: This component is responsible for storing vectors that represent the text chunks used to assist the Language Model (LLM) in providing accurate answers. In our case, we used PostgreSQL with the PGVector extension for this purpose.

3. **Embedder**: The Embedder is responsible for converting text into vectors. We utilized the *all-MiniLM-L6-v2* model from HuggingFace for this task.

4. **LLM**: We incorporated a pretrained Language Model from HuggingFace, such as *GPT-2* or *Flan-T5*, to be fed with prompts.

5. **Orchestrator**: Langchain is a well-established Python library with numerous integrations that facilitate the creation of LLM applications. Additionally, we implemented a strategy pattern, allowing for easy interchangeability of the LLM and/or the table used for storing embeddings.

6. **Promt template**: In order to help the LLM model the result that we want, we will use the following template so that the LLM knows has an idea of what part is context and what is the question:
    ```python
    """Based on the following context:.\n
      {context}.\n\n
      
      Answer adding an explanation, this question:\n\n
      
      {question}"""
    ```

## Evaluation


To evalute a RAG System let's use the Framework RAGAS and the following metrics.

- **Faithfulness**: This measures the factual consistency of the generated answer against the given context. It is calculated from **answer** and **retrieved context**. The answer is scaled to $(0,1)$ range. Higher the better. The generated answer is regarded as faithful if all the claims that are made in the answer can be inferred from the given context.

- **Answer Relevance**: The evaluation metric, Answer Relevance, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the **question**, the **context** and the **answer**.
- **Context Precision**:Evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the **question**, **ground_truth** and the **contexts**.
- **Context Relevancy**: This metric gauges the relevancy of the retrieved context, calculated based on both the **question** and **contexts**.
- **Context Recall**: Measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the **ground truth** and the **retrieved context**.
- **Answer Correctness**: The assessment of Answer Correctness involves gauging the accuracy of the generated **answer** when compared to the **ground truth**. This evaluation relies on the **ground truth** and the **answer**.


Let's pay special attention to following 3 metrics, that has became a widely used triad to evaluate RAG systems:
- Context Relevance
- Answer Relevance
- Faithfulness

With these metrics, we are ready to perform a hyperparameter tuning, to choose the best RAG structure for the current problem.

# How to run the App Locally?


1. Build a custom postgres image


```
cd src/database
```

```
docker build -t vector-postgres-image .

```

2. Run the container
```bash
docker run -d \
  --name postgres-llm-container \
  -e POSTGRES_DB=llm_aws_docs \
  -e POSTGRES_USER=llmadmin \
  -e POSTGRES_PASSWORD=llmpassword \
  -e POSTGRES_HOST=localhost \
  -e POSTGRES_PORT=5432 \
  -p 5432:5432 \
  vector-postgres-image
```

3. Install all the python libraries needed for the project.

```bash
pip install -r dev_requirements.txt
```

4. Create a table of vectors with the documents in a folder "/path/to/data/".

```bash
python src/database/database.py --env dev --table_name mitabla /path/to/data/
```
python src/database/database.py --env dev --table_name clementine_embeddings /data/

5. Run the front-end

```
streamlit run frontend.py
```

# AWS Deployment

In the future, all the components of this app can be easily deployed to AWS. Below is a picture of simple arquitecture, which use Ec2 intances to host the front-end and the RagApp (back-end), an RDS to create a PostgreSql and an S3 to store the documents.
![RAG Image](images/RAG-Arq.png)




