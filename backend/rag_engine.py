from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI


def build_vector_store(text):

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_texts(chunks, embeddings)

    vector_store.save_local("vector_db")

    return vector_store


def ask_question(question):

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.load_local(
        "vector_db",
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs = vector_store.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Use the following research paper context to answer the question.

Context:
{context}

Question:
{question}
"""

    llm = ChatOpenAI(model="gpt-4o-mini")

    response = llm.invoke(prompt)

    return response.content