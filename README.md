# MultiPDF Chat App

# Introduction
<p>The Gemini Pro Pdf Chatbot is a Python application that allows you to chat with multiple PDF documents. You can ask questions about the PDFs using natural language, and the application will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.</p>

# How  It Works
<p>The application follows these steps to provide responses to your questions:</p>
1. <b>PDF Loading:</b> The app reads multiple PDF documents and extracts their text content.<br>
2. <b>Text Chunking:</b> The extracted text is divided into smaller chunks that can be processed effectively.<br>
3. <b>Language Model:</b> The application utilizes a language model to generate vector representations (embeddings) of the text chunks.<br>
4. <b>Similarity Matching:</b> When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.<br>
5. <b>Response Generation:</b> The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.<br>

# Dependencies and Installation
 <p>To install the Gemini Pro Pdf Chatbot, please follow these steps:</p>
   1. Clone the repository to your local machine.

2. Install the required dependencies by running the following command:

    ```
    pip install -r requirements.txt
    ```

3. Obtain an API key from Google (GCP Console) and add it to the `.env` file in the project directory.

    ```shell
    GOOGLE_API_KEY=your_secret_api_key
    ```

## Usage

To use the Gemini Pro Pdf Chatbot, follow these steps:

1. Ensure that you have installed the required dependencies and added the Google API key to the `.env` file.

2. Run the `app.py` file using the Streamlit CLI. Execute the following command:

    ```
    streamlit run app.py
    ```

3. The application will launch in your default web browser, displaying the user interface.

4. Load multiple PDF documents into the app by following the provided instructions.

5. Ask questions in natural language about the loaded PDFs using the chat interface.

## Contributing
<p>We welcome contributions from the community to improve and expand Gemini Pro Pdf Chatbot. If you have suggestions or would like to contribute, please create a pull request or open an issue.</p>

## License
<p>Gemini Pro Pdf Chatbot is licensed under the MIT License.</p>

## Acknowledgments
<p>We would like to express our gratitude to the open-source community (google, Langchain) for the tools and libraries that make this project possible. Special thanks to the creators of Streamlit and Langchain for their outstanding work.</p>