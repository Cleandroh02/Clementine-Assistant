import pydantic

from typing import Optional, Dict

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores.base import VectorStore
from langchain.schema.embeddings import Embeddings
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from pydantic import BaseModel
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForCausalLM
    )

from src.config import get_settings


class RagApp(BaseModel):
    table_name: str = "clementine_embeddings"
    llm: HuggingFacePipeline = None
    llm_model_name: str = "google/flan-t5-large"
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    db: VectorStore = None
    embeddings: Embeddings = None
    template: Optional[str] = """Based on the following context:.\n {context}.\n\n Answer adding an explanation, this question:\n\n {question}"""
    env: str = "dev"

    class Config:
        arbitrary_types_allowed = True

    @pydantic.validator("db", pre=True)
    @classmethod
    def is_db_valid(cls, value):
        """
        Validates the database used to store the documents.

        Parameters:
        -----------
        value: VectorStore | str
            The database to be validated.

        Returns:
        --------
        value: VectorStore | str
            The validated database.
        """
        if isinstance(value, str):
            if value not in ["PGVector"]:
                raise ValueError(
                    f"The value {value} is not a valid value for the database."
                )
        elif isinstance(value, VectorStore):
            implemented_vectorstores = [PGVector]
            if not isinstance(value, implemented_vectorstores):
                raise NotImplementedError(
                    f"The Vectorstore {value} have not been implemented yet."
                )

        return value

    def __init__(self, *args, **kwargs):
        """
        Initializes the RagApp class.

        Parameters:
        -----------
        args: Any
            Positional arguments.
        kwargs: Any
            Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        settings = get_settings(self.env)

        self._set_db(settings)
        self._set_llm()

    def _set_db(self, settings):
        CONNECTION_STRING = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=settings.POSTGRESDB_URL,
            port=int(settings.POSTGRESDB_PORT),
            database=settings.POSTGRESDB_NAME,
            user=settings.POSTGRESDB_USER,
            password=settings.POSTGRESDB_PASSWORD,
        )
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embeddings_model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )
        self.db = PGVector(
            embedding_function=self.embeddings,
            collection_name=self.table_name,
            connection_string=CONNECTION_STRING,
        )

    def _set_llm(self):
        if self.llm_model_name not in [
            "gpt2",
            "google/flan-t5-large",
        ]:
            self.llm_model_name = "google/flan-t5-large"
        
        assert self.llm_model_name == "google/flan-t5-large"

        tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)

        if self.llm_model_name == "gpt2":
            print(self.llm_model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model_name)
            model = AutoModelForCausalLM.from_pretrained(self.llm_model_name)
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512
            )

        

        elif "google/flan-t5" in self.llm_model_name:
            model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model_name)
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=512,
            )

        self.llm = HuggingFacePipeline(
            pipeline=pipe,
            model_kwargs={"temperature": 0},
        )

    def answer_question(self, question: str, top_k: int = 2) -> Dict:
        """Answer a question based on an LLM model and a retriever.

        Args:
            question (str): Custom question in English
            top_k (int, optional): Number of documents to retrieve.
                Defaults to 5.

        Returns:
            Dict: Dictionary containing the answer and the top-k most
                similar documents documents
        """
        retriever = self.db.as_retriever(
            search_type="mmr", search_kwargs={"k": top_k, "lambda_mult": 0.5}
        )

        promt = PromptTemplate(
            template=self.template, input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": promt},
            return_source_documents=True,
        )
        return qa_chain(question)
