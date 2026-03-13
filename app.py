
import os
import time
import tempfile
import base64
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")    
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]= "RAG Document Q&A"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")



from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from openai import OpenAI
from gtts import gTTS


MAX_QUESTIONS_PER_SESSION = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL =  "llama-3.1-8b-instant"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY in .env")
    st.stop()

llm = ChatGroq(groq_api_key=groq_api_key, model=LLM_MODEL)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# OpenAI client for Whisper (optional — voice features disabled if missing)
whisper_available = bool(openai_api_key)
if whisper_available:
    openai_client = OpenAI(api_key=openai_api_key)



def transcribe_audio(audio_bytes: bytes) -> str:
    """Transcribe audio bytes using OpenAI Whisper API."""
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "recording.wav"
    transcript = openai_client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text


def text_to_speech(text: str) -> bytes:
    """Convert text to speech using gTTS and return audio bytes."""
    tts = gTTS(text=text, lang="en", slow=False)
    audio_buffer = BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


def autoplay_audio(audio_bytes: bytes):
    """Auto-play audio in Streamlit using HTML audio tag."""
    b64 = base64.b64encode(audio_bytes).decode()
    html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(html, unsafe_allow_html=True)



st.set_page_config(page_title="Conversational RAG", page_icon="📄", layout="wide")
st.title("📄 Conversational RAG — PDF Chat with Voice")
st.caption(
    "Upload PDFs, ask questions via text or voice • "
    "Powered by LangChain + FAISS + Groq + Whisper"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_count" not in st.session_state:
    st.session_state.question_count = 0
if "store" not in st.session_state:
    st.session_state.store = {}
if "retriever_ready" not in st.session_state:
    st.session_state.retriever_ready = False
if "query_logs" not in st.session_state:
    st.session_state.query_logs = []


with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDFs here", type="pdf", accept_multiple_files=True
    )

    session_id = st.text_input("Session ID", value="default_session")

    # Voice settings
    st.divider()
    st.header("🎙️ Voice Settings")
    voice_output_enabled = st.toggle("Enable voice responses", value=False)

    if not whisper_available:
        st.info("Add OPENAI_API_KEY to .env to enable voice input (Whisper)")

    # Rate limit display
    st.divider()
    remaining = MAX_QUESTIONS_PER_SESSION - st.session_state.question_count
    st.metric("Questions remaining", f"{remaining}/{MAX_QUESTIONS_PER_SESSION}")

    if remaining <= 0:
        st.warning("Free demo limit reached. Clone the repo to run unlimited.")

    # Index button
    if uploaded_files and st.button("📥 Index Documents", type="primary"):
        with st.spinner("Chunking & indexing..."):
            documents = []
            for uploaded_file in uploaded_files:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                documents.extend(docs)
                os.unlink(tmp_path)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(
                documents=splits, embedding=embeddings
            )
            st.session_state.retriever = vectorstore.as_retriever(
                search_kwargs={"k": 4}
            )
            st.session_state.retriever_ready = True
            st.session_state.num_chunks = len(splits)

        st.success(
            f"Indexed {len(splits)} chunks from {len(uploaded_files)} PDF(s)"
        )

    # Per-query eval metrics
    if st.session_state.query_logs:
        st.divider()
        st.header("📊 Query Performance")
        for i, log in enumerate(st.session_state.query_logs[-5:], 1):
            with st.expander(f"Q{log['query_num']}: {log['question'][:40]}..."):
                st.write(f"⏱️ Latency: **{log['latency_ms']}ms**")
                st.write(f"📄 Chunks retrieved: **{log['chunks_retrieved']}**")
                st.write(f"📏 Response length: **{log['response_length']} chars**")


if st.session_state.retriever_ready:
    retriever = st.session_state.retriever

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given a chat history and the latest user question which might "
            "reference context in the chat history, formulate a standalone "
            "question which can be understood without the chat history. "
            "Do NOT answer the question, just reformulate it if needed "
            "and otherwise return it as is.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a helpful assistant for answering questions about documents. "
        "Use the following retrieved context to answer the question. "
        "If you don't know the answer, say so. Keep answers concise and "
        "well-structured — use 3-5 sentences.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session: str) -> BaseChatMessageHistory:
        if session not in st.session_state.store:
            st.session_state.store[session] = ChatMessageHistory()
        return st.session_state.store[session]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

   
    def process_question(question: str):
        """Run RAG chain, track metrics, display response."""
        if st.session_state.question_count >= MAX_QUESTIONS_PER_SESSION:
            st.error("Free demo limit reached. Clone the repo to run unlimited.")
            return

        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate response with timing
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                try:
                    response = conversational_rag_chain.invoke(
                    {"input": question},
                    config={"configurable": {"session_id": session_id}},
                    )
                except Exception as e:
                    if "rate_limit" in str(e).lower() or "429" in str(e):
                        st.error("⚠️ API rate limit reached. Please try again in a few minutes.")
                        return
                    raise e
                latency_ms = int((time.time() - start_time) * 1000)

                answer = response["answer"]
                chunks_retrieved = len(response.get("context", []))

                st.markdown(answer)

                # Voice output
                if voice_output_enabled:
                    with st.spinner("Generating voice..."):
                        audio_bytes = text_to_speech(answer)
                        autoplay_audio(audio_bytes)

        # Save message
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.question_count += 1

        # Log query metrics
        st.session_state.query_logs.append({
            "query_num": st.session_state.question_count,
            "question": question,
            "latency_ms": latency_ms,
            "chunks_retrieved": chunks_retrieved,
            "response_length": len(answer),
        })

   
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    
    if whisper_available:
        audio_value = st.audio_input("🎙️ Or ask with your voice")
        if audio_value:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(audio_value.getvalue())
            st.info(f"🎙️ You said: *{transcript}*")
            process_question(transcript)

   
    if user_input := st.chat_input("Ask a question about your documents..."):
        process_question(user_input)

else:
    st.info("👈 Upload PDFs and click **Index Documents** to get started.")










