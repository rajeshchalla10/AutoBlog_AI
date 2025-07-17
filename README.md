# AutoBlog AI- AI Assistant to generate Blog from Document

<img src="Demo/01 AutoBlog AI.png" width="400"/> <img src="Demo/02 AutoBlog AI.png" width="400"/><img src="Demo/03 AutoBlog AI.png" width="400"/> <img src="Demo/04 AutoBlog AI.png" width="400"/> 

## Overview

This project AutoBlog-AI is built using Python, Flask, Langchain, and RAG (Retrieval-Augmented Generation) based of Large Language Model (LLM). This project allows users to upload any document (PDF, DOCX, TXT), and automatically generates a summarized blog post using advanced AI techniques.

## Features

- **Document Upload**: Users can upload documents via a simple web interface.
- **Automated Blog Generation**: Converts document content into a well-structured blog post.
- **Text Chunking and Storage**: Splits the extracted document into manageable chunks using RecursiveCharacterTextSplitter and stores them in a FAISS Vector Database.
- **LLM-Powered Summarization**: Uses Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs) for high-quality output.
- **Contextual Understanding**: Employs LangChain for enhanced document comprehension and chaining.
- **Flask Web Interface**: Provides a simple web interface for users to interact with the assistant.

## Tech Stack

- **Python**: Core programming language.
- **Flask**: A lightweight web framework used for building the application's interface.
- **RAG (Retrieval-Augmented Generation)**: Integrates document retrieval with LLMs for better summarization.
- **LLM (Large Language Models)**: Powers the natural language understanding and generation.
- **LangChain**: Framework for chaining LLMs and retrieval modules.

## How It Works

1. **User uploads a document** via the web interface.
2. **Document is processed** and split into meaningful chunks.
3. **RAG pipeline retrieves** relevant information from the document.
4. **LLM generates** a blog post draft based on the document's content.
5. **Final blog post** is presented in markup language for users to use or further edit.

## Getting Started

### Prerequisites
- Check the requirements.txt file to know the required libraries to install.


### Installation

```bash
git clone https://github.com/rajeshchalla10/AutoBlog_AI.git
cd AutoBlog_AI
pip install -r requirements.txt
```
### Running the App

```bash
python test_app.py
```

Open [http://localhost:5000](http://localhost:5000) (or supported port by the application).


## Project Structure

```
document-to-blog-assistant/
│
├── test_app.py            # Flask server and endpoints
├── templates/             # HTML templates for web UI
├── static/                # Static files (CSS, JS)
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License.


