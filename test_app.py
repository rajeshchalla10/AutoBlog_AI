from flask import Flask, render_template, request
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # Or your preferred vector database
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv
import os
from flask import Flask, render_template, request, flash
from werkzeug.utils import secure_filename
import markdown

# Load environment variables
load_dotenv() 


app = Flask(__name__)

# Global variables for storing video information
vector_store = None
chain = None
target_audience = None
chat_history = []
user_keywords_history = []  # Initialize user keywords history




# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'  # Folder to save uploaded files
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'}  # Allowed file extensions

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Set a secret key for flashing messages (important for production)
app.secret_key = '1920931030'
# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    """Checks if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




# Initialize components
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")


def format_docs(retrived_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrived_docs)
    return context_text



@app.route("/", methods=["GET", "POST"])
def index():
    return render_template('index.html')


@app.route("/upload", methods=["GET", "POST"])
def upload_document():
    global vector_store, chain,target_audience,chat_history,user_keywords_history
    
    """Handles the document upload and form data submission from index.html."""
    file = request.files['document']

    if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)


    # Retrieve other form data
    keyword = request.form['keyword']
    target_audience = request.form['audience']

    # Store the keyword in history
    user_keywords_history = []
    user_keywords_history.append(keyword)  
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    print(len(docs))
    print(f"Keyword: {keyword}")
    print(f"Audience: {target_audience}")
     # --- Remove the file after processing ---
    try:
        os.remove(file_path)
        print(f"Successfully deleted file: {file_path}")
    except OSError as e:
                    print(f"Error deleting file {file_path}: {e}")
                    flash(f"Error deleting file {filename}. Check server logs.")

                    
    #splitting text
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, embeddings)

    #retriver
    retriever = vector_store.as_retriever(search_type="similarity", search_kwards={"k": 10})
    #print(result)




    #blog generation
    template = """
    Based on the information provided below, craft a comprehensive blog post using the supplied Context.
    
    Requirements:
    The blog must be structured to rank highly for the target keywords: {keyword}.
    Maintain an engaging and informative tone suitable for a {target_audience} audience. 
                    
    Instructions:
    The blog should be properly and beautifully formatted using markdown.
    Create an SEO-optimized, compelling blog title that incorporates the target keyword and appeals to readers—catchy but not exaggerated.
    Clearly mark the title, section headings, and sub-headings using markdown conventions.
    Organize the blog into sections, each containing at least two subsections.
    Each subsection should have a minimum of two paragraphs.
    Cover all specified aspects in each section and provide detailed, informative content for each subtopic.
    Ensure logical flow and coherence throughout the blog for easy readability.
    Where appropriate, enrich the post with examples, case studies, or insights to deepen understanding.
    For topics involving data privacy, bias, or responsible use, include thoughtful discussions on ethical considerations.
    Conclude with a forward-looking perspective and a summary of key points.
    Maintain professional, standard markdown formatting throughout.
    Make the blog engaging and relatable, using real-world examples and providing thorough information.
    You are a professional blog post writer and SEO expert.

    Context: {context}
    Blog: 
    """

    #prompt
    prompt = PromptTemplate.from_template(template=template)

    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke({"context": retriever | RunnableLambda(format_docs), 
        "keyword": keyword, 
        "target_audience": target_audience})

    html_summary = markdown.markdown(response)

    print(response)
    chat_history.append((template, response))


    # Render the blog post in the blog.html template
    return render_template('blog.html',summary=html_summary)





@app.route('/submit-context', methods=['GET', 'POST'])
def submit_context():
    global target_audience,chat_history,chain, user_keywords_history
    
    user_question = request.form['contextText']
    
    
    # Store the user question in history
    user_keywords_history.append(user_question)  
 
    response = chain.invoke({"context": chat_history, 
        "keyword": user_keywords_history,
        "target_audience": target_audience})

    html_summary = markdown.markdown(response)



    print('chat history:', chat_history)
    print('user keywords history:', user_keywords_history)
    print('user keywords history:', user_keywords_history)
    print(response)


    chat_history.append((user_question, response))

    

    return render_template('blog.html',summary=html_summary)






if __name__ == '__main__':
    app.run(debug=True)