import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

#Set the API key
api_key = os.getenv("GOOGLE_API_KEY")

#Load the models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

#Configure Chroma as a retriever with top_k=3
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing Chroma vectorstore
vectordb = Chroma(
    persist_directory="chroma/",
    embedding_function=embedding_function
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


### Contextualize question ###
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)



#Create the retrieval chain
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# context = retriever.get_relevant_documents(user_input)
# print("Retrieved Context:", context)

#Print the answer to the question
# print(response["answer"])
chat_history = []
while True:
    question = input("You: ")
    response = rag_chain.invoke({"input": question,"chat_history": chat_history})
    print("Bot:", response["answer"])
    chat_history.extend([HumanMessage(content=question), response["answer"]])