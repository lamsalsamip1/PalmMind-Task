## RAG Agent Based Chatbot to ask queries and make bookings.

#Steps to run the chatbot:

1. Create a virtual environment
   ``` python -m venv env ```
2. Install the dependencies.
   ``` pip install -r requirements.txt ```
3. Recreate the vector database.
   ``` python -m create_db.py```
4. Run chat.py to chat from the terminal
   ``` python -m chat.py```

Bookings are stored in booking_details.txt file.
