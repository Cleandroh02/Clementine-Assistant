import os
import re
import click

from pydantic import BaseSettings
from langchain.vectorstores import PGVector
from psycopg2 import connect
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import sys # noqa
sys.path.append('.') # noqa
from src.config import get_settings


def get_database(settings: BaseSettings, table_name: str) -> PGVector:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    CONNECTION_STRING = PGVector.connection_string_from_db_params(
        driver="psycopg2",
        host=settings.POSTGRESDB_URL,
        port=int(settings.POSTGRESDB_PORT),
        database=settings.POSTGRESDB_NAME,
        user=settings.POSTGRESDB_USER,
        password=settings.POSTGRESDB_PASSWORD,
    )
    db = PGVector(
        embedding_function=embeddings,
        collection_name=table_name,
        connection_string=CONNECTION_STRING,
    )

    return db


def create_vector_extension(settings) -> None:
    connection = connect(
        host=settings.POSTGRESDB_URL,
        port=int(settings.POSTGRESDB_PORT),
        database=settings.POSTGRESDB_NAME,
        user=settings.POSTGRESDB_USER,
        password=settings.POSTGRESDB_PASSWORD,
    )

    cursor = connection.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    connection.commit()
    connection.close()


def process_md_file(
    file: str, text_splitter: RecursiveCharacterTextSplitter, db: PGVector
) -> None:
    loader = TextLoader(file, encoding="utf-8")
    pattern = re.compile(r"[^a-zA-Z0-9\s-]|(http[s]?://\S+)")
    doc_ = loader.load()
    assert len(doc_) == 1
    doc_[0].page_content = re.sub(pattern, " ", doc_[0].page_content)
    texts = text_splitter.split_documents(doc_)
    db.add_documents(texts)


@click.command()
@click.argument("folder_path", type=click.Path(exists=True))
@click.option("--env",
              default="env",
              help="Environment name (default: env)",
              type=str)
@click.option(
    "--table_name",
    default="test_db",
    help="Collection name (default: test_db)",
    type=str,
)
def process_md_files(folder_path, env, table_name):
    """Process a list of Markdown files in the specified folder."""
    click.echo(f"Processing Markdown files in folder: {folder_path}")

    settings = get_settings(env=env)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=300
        )

    create_vector_extension(settings)

    db = get_database(settings, table_name)
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                click.echo(f"Processing file: {file}")
                process_md_file(file_path, text_splitter, db)


if __name__ == "__main__":
    process_md_files()
