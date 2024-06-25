import os
# import chromadb
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path=".langchain.db"))

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')


# Question and Answer pipeline
# 1. Prepare the doc (once)
#     Load data into langchain docs
#     Split the doc into chunks
#     Embed chunks into vectors
#     save chunks and embeddings to vector db


# load private local file
# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


def load_from_wikipedia(query, lang='en', load_max_docs=2):
    from langchain_community.document_loaders import WikipediaLoader
    loader = WikipediaLoader(query=query, lang=lang, load_max_docs=load_max_docs)
    data = loader.load()
    return data


# chunking data

# 2. Search (once)
#     Embed question
#     Using question and chunk embeddings, rank vectors by similarity to question embedding. Closer is more related

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks


# for determining embedding cost
def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding cost: ${total_tokens / 1000 * 0.0004:.6f}')


def insert_or_fetch_embeddings(index_name, chunks):
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    pc = pinecone.Pinecone()
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings...', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        print(f'Creating index {index_name} and embeddings...', end='')
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=PodSpec(environment='gcp-starter')
        )
    vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    print('Ok')
    return vector_store


def delete_pinecone_index(index_name='all'):
    import pinecone
    pc = pinecone.Pinecone()
    if index_name == 'all':
        indexes = pc.list_indexes().names()
        print('Deleting all indexes...')
        for index in indexes:
            pc.delete_index(index)
        print('Ok')
    else:
        print(f'Deleting index {index_name}...', end='')
        pc.delete_index(index_name)
        print('Ok')


def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Create a Chroma vector store using the provided text chunks and embedding model,
    # configuring it to save data to the specified directory
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    # try:
    #     vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)
    # except AttributeError as err:
    #     print(err)

    return vector_store  # Return the created vector store


def load_embeddings_chroma(persist_directory='./chroma_db'):
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Instantiate the same embedding model used during creation
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)

    # Load a Chroma vector store from the specified directory, using the provided embedding function
    vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

    return vector_store  # Return the loaded vector store

# Loading the pdf document into LangChain
# data = load_document('files/rag_powered_by_google_search.pdf')
#
# # Splitting the document into chunks
# chunks = chunk_data(data)
#
# # Creating a Chroma vector store using the provided text chunks and embedding model (default is text-embedding-3-small)
# vector_store = create_embeddings_chroma(chunks)
#
#
# # # Instantiate a ChatGPT LLM (temperature controls randomness)
# llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
#
# # # Configure vector store to act as a retriever (finding similar items, returning top 5)
# retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})
#
# # # Create a memory buffer to track the conversation
# memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)


# custom prompt
system_template = r'''
Use the following pieces of context to answer the user's question.
If you don't find the answer in the provided context, respond with "Sorry, I don't know"
------------
Context: ```{context}```
'''

user_template = '''
Question: ```{question}```
Chat History: ```{chat_history}```
'''

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(user_template)
]

# qa_prompt = ChatPromptTemplate.from_messages(messages)
#
# crChain = ConversationalRetrievalChain.from_llm(
#     llm=llm,  # Link the ChatGPT LLM
#     retriever=retriever,  # Link the vector store based retriever
#     memory=memory,  # Link the conversation memory
#     chain_type='stuff',  # Specify the chain type
#     combine_docs_chain_kwargs={'prompt':qa_prompt},
#     verbose=False  # Set to True to enable verbose logging for debugging
# )

# 3. Ask(once)
#     insert question and relevant chunks into message for gpt model
#     return model's answer


def ask_question(q, chain):
    result = chain.invoke({'question': q})
    return result


# print(qa_prompt)
#
# db = load_embeddings_chroma()
# question = 'How many pairs of questions and answers had the StackOverflow dataset?'
# result = ask_question(question, crChain)
# print(result)

def ask_and_get_answer(vector_store, q):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=retriever)

    answer = chain.invoke(q)
    return answer


# db = load_embeddings_chroma()
# q = 'How many pairs of questions and answers had the StackOverflow dataset?'
# answer = ask_and_get_answer(vector_store, q)
# print(answer)


def ask_from_document(question):
    delete_pinecone_index()
    data = load_document('files/us_constitution.pdf')

    chunks = chunk_data(data)

    index_name = 'askadocument'
    vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

    answer = ask_and_get_answer(vector_store, question)
    print(answer)


def ask_from_wikipedia(question, topic):
    delete_pinecone_index()
    data = load_from_wikipedia(topic, 'en')
    chunks = chunk_data(data)

    index_name = 'wikipedia'
    vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)

    answer = ask_and_get_answer(vector_store, question)
    print(answer)


question = 'When ws the last time that the Chicago Cubs won the World Series?'
topic = 'Chicago Cubs'
print(ask_from_wikipedia(question, topic))


# 3. Ask until quit
# i = 1
# print('Write "Quit" or "Exit" to end the application')
#
# while True:
#     question = input(f'Question #{i}: ')
#     i += 1
#     if question.lower() in ['quit', 'exit']:
#         print('Quitting the application')
#         time.sleep(2)
#         break
#
#     answer = ask_and_get_answer(vector_store, question)
#     print(f'\nAnswer: {answer}')
#     print(f'\n {"-" * 50} \n')
#
