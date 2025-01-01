import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage
from validate import validate_date, validate_email
from langchain.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent


# Set the API key
api_key = os.getenv("GOOGLE_API_KEY")

# Load the models
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

# Configure Chroma as a retriever with top_k=3
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load the existing Chroma vectorstore
vectordb = Chroma(
    persist_directory="chroma/",
    embedding_function=embedding_function
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# History-aware retriever setup
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

# Create the retrieval chain
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

# Booking tool definition
@tool
def booking_tool() -> str:
    """
    This tool books a trekking package. It asks the user for all the information and validates it.
    If the booking is successful, it returns 'Booking successful'. If the booking fails, it returns 'Booking failed'.
    """
    booking_details = {}

    # Ask for trekking destination
    trekking_destination = input("""
                                 Please enter your trekking destination: 
                                 Available destinations: Langtang, ABC, Ama Yangri, Shey Phoksundo
                                 """)
    booking_details['destination'] = trekking_destination

    # Ask for date and validate format
    while True:
        trekking_date = input("Please enter the date of your trek (YYYY-MM-DD): ")
        if validate_date(trekking_date):
            booking_details['date'] = trekking_date
            break
        else:
            print("Invalid date format. Please enter a valid date.")

    # Ask for name
    booking_details['name'] = input("Please enter your name: ")

    # Ask for email and validate
    while True:
        email = input("Please enter your email address: ")
        if validate_email(email):
            booking_details['email'] = email
            break
        else:
            print("Invalid email format. Please try again.")

    # Ask for phone number
    while True:
        phone = input("Please enter your phone number: ")
        if phone.isdigit() and len(phone) == 10:
            booking_details['phone'] = phone
            break
        else:
            print("Invalid phone number. Please enter a 10-digit number.")

    # Save booking details to a file
    with open("booking_details.txt", "a") as f:
        f.write(str(booking_details) + "\n")
    
    return "Booking successful"

@tool
def rag_chain_tool(query: str, chat_history: list[str]) -> str:
    """ Tool to invoke the RAG chain with the given query and chat history. """
    return rag_chain.invoke({"input": query, "chat_history": chat_history})["answer"]
# Define tools
tools = [
    booking_tool,
    rag_chain_tool
]

#Define agent prompt
agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a smart assistant capable of answering questions related to trekking and managing bookings. If the user wants any information about trekking, invoke the rag_chain_tool to answer it. If the user wants to book a trek or a trip or anything, invoke the booking_tool. You must invoke one of the tools."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
# agent_prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", "You are a helpful assistant that manages trekking bookings and can answer questions about trekking. You will always either provide an answer from the RAG system or initiate the booking process."),
#         MessagesPlaceholder("chat_history"),
#         ("human", "{input}"),
#         MessagesPlaceholder("agent_scratchpad")
#     ]
# )

# Create agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Chat interface
chat_history = []
while True:
    user_input = input("You: ")
    response = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    print("Bot:", response['output'])

    # Update chat history only if the booking tool wasn't invoked
    if "booking_tool" not in response.get('tool_used', ''):  # Check if 'booking_tool' is invoked
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response['output']))
    else :
        #Tool invoking was having problems after invoking a booking once. So, clearing the chat history after a successful booking.
        chat_history.clear()  
   
