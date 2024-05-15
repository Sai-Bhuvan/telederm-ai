import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = load_model("my_model.h5")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

    return response["output_text"]

def predict_image(image):
    img = cv2.resize(image, (224, 224))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values (if your model expects normalization)

    # Make prediction
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)

    # Interpret predictions (print or process them as needed)
    # For example, if 'predictions' contains probabilities for each class:
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_class, confidence

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with TelidermAI to get Your Doubts Cleared💁")

    # Initialize conversation history as an empty list
    conversation_history = []

    user_question = st.text_input("Ask a Question related to the diseases")

    if user_question:
        # Get response from user input and add to conversation history
        response = user_input(user_question)
        conversation_history.append(("User:", user_question))
        conversation_history.append(("TelidermAI:", response))

    # Display conversation history
    # st.subheader("Conversation History")
    # for role, text in conversation_history:
    #     st.write(f"{role} {text}")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Creating Embeddings..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

        st.title("Image Prediction")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            image = np.array(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(image, -1)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            if st.button('Predict'):
                with st.spinner("Predicting..."):
                    predicted_class, confidence = predict_image(image)
                st.write(f"Predicted Class Index: {predicted_class}")
                st.write(f"Confidence: {confidence}")


if __name__ == "__main__":
    main()
