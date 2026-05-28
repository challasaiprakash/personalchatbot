import os
import re
import streamlit as st
from datetime import datetime

# --- Google Sheets ---
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

# --- LangChain Core Components ---
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

# --- LangChain Document Indexing ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- LangChain Vector Storage & Open-Source Embeddings ---
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain_community.vectorstores import Chroma

# ─── Page Config & Header ────────────────────────────────────────────────────
st.set_page_config(page_title="AskSai", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
    .asksai-header {
        text-align: center;
        padding: 0.6rem 0 0.2rem 0;
    }
    .asksai-header h1 {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C63FF, #48CAE4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .asksai-header p {
        color: #888;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    </style>
    <div class="asksai-header">
        <h1>🧠 AskSai</h1>
        <p>Your direct line to Sai's professional profile</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─── Constants ────────────────────────────────────────────────────────────────
DATA_DIR = "data"
DEFAULT_RESUME_PATH = os.path.join(DATA_DIR, "resume_and_projects.txt")
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

OUT_OF_SCOPE_PHRASES = [
    "goes beyond what i currently have on file",
    "beyond what i currently have on file",
    "not available in sai's profile",
    "flag this for him",
    "sai will be happy to address it personally",
    "reach out to him directly",
    "not in the provided context",
    "outside the scope",
    "not available in the candidate",
    "not contained in the context",
    "not mentioned in the context",
    "not included in the context",
    "no information about",
    "no information on",
    "don't have information",
    "do not have information",
    "doesn't appear in",
    "does not appear in",
    "cannot find",
    "can't find",
    "unable to find",
    "not found in",
    "not present in",
    "context does not",
    "context doesn't",
    "provided context",
    "i'm sorry, but",
    "i am sorry, but",
    "unfortunately",
    "not specified",
    "not mentioned",
    "not discussed",
    "not covered",
]

OUT_OF_SCOPE_REPLY = (
    "That's a great question! This goes beyond what I currently have on file for Sai. "
    "I'll flag this for him — Sai will be happy to address it personally in the future. "
    "Feel free to reach out to him directly in the meantime!"
)

SHEET_HEADERS = ["#", "Timestamp", "Question", "Status"]


# ─── Config Helper (st.secrets ONLY) ─────────────────────────────────────────
def get_config(key: str, default=None):
    try:
        val = st.secrets[key]
        return str(val) if isinstance(val, bool) else val
    except (KeyError, FileNotFoundError):
        return default


def get_service_account_info():
    """
    Returns the GCP service account dict from st.secrets["service_account"].
    FIX: Replaces literal \\n with real newlines in private_key.
    Streamlit Cloud stores TOML secrets as literal \\n which breaks Google auth.
    """
    try:
        info = dict(st.secrets["service_account"])
        if info:
            # ✅ CRITICAL FIX: Streamlit Cloud TOML stores \n as literal characters.
            # Google auth needs real newline characters in the private key.
            if "private_key" in info:
                info["private_key"] = info["private_key"].replace("\\n", "\n")
            return info, None
    except (KeyError, FileNotFoundError):
        pass

    return None, (
        "Google service account credentials not found. "
        "Make sure [service_account] is present in your Streamlit secrets."
    )


# ─── Google Sheets Logger ─────────────────────────────────────────────────────
def get_sheet():
    """Authenticate and return (worksheet, error_string_or_None)."""
    if not GSPREAD_AVAILABLE:
        return None, (
            "gspread / google-auth not installed. "
            "Add 'gspread' and 'google-auth' to your requirements.txt and redeploy."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    service_info, err = get_service_account_info()
    if err:
        return None, err

    try:
        creds = Credentials.from_service_account_info(service_info, scopes=scopes)

        # ✅ FIX: Use non-deprecated gspread client creation.
        # gspread.authorize() is deprecated in newer versions used on Streamlit Cloud.
        try:
            client = gspread.Client(auth=creds)
            client.session = requests.Session() if False else client.session  # noqa
        except Exception:
            # fallback for older gspread versions
            client = gspread.authorize(creds)

        sheet_id = (get_config("GOOGLE_SHEET_ID") or "").strip()
        if not sheet_id:
            return None, "GOOGLE_SHEET_ID is missing from Streamlit secrets."

        spreadsheet = client.open_by_key(sheet_id)
        worksheet = spreadsheet.sheet1

        existing = worksheet.get_all_values()
        if not existing or existing[0] != SHEET_HEADERS:
            worksheet.clear()
            worksheet.append_row(SHEET_HEADERS)
            worksheet.format("A1:D1", {"textFormat": {"bold": True}})
            worksheet.freeze(rows=1)

        return worksheet, None

    except Exception as exc:
        return None, f"Google Sheets connection failed: {exc}"


def log_out_of_scope(question: str):
    """Append an out-of-scope question to Google Sheets. Returns error string or None."""
    worksheet, err = get_sheet()
    if err:
        return err
    try:
        existing = worksheet.get_all_values()
        row_num = len(existing)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        worksheet.append_row([row_num, timestamp, question, "Pending"])
        return None
    except Exception as exc:
        return str(exc)


# ─── Resume Helpers ───────────────────────────────────────────────────────────
def get_resume_path():
    if os.path.exists(DEFAULT_RESUME_PATH):
        return DEFAULT_RESUME_PATH
    if not os.path.isdir(DATA_DIR):
        return None
    text_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.lower().endswith(".txt")
    ]
    return text_files[0] if text_files else None


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_project_docs(text, source):
    pattern = re.compile(
        r"(PROJECT\s+\d+:\s.*?)(?=\nPROJECT\s+\d+:|\n={5,}|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    project_docs = []
    for match in pattern.finditer(text):
        project_text = match.group(1).strip()
        number_match = re.search(r"PROJECT\s+(\d+):", project_text, re.IGNORECASE)
        project_number = int(number_match.group(1)) if number_match else None
        project_docs.append(
            Document(
                page_content=project_text,
                metadata={
                    "source": source,
                    "section": "project",
                    "project_number": project_number,
                },
            )
        )
    return project_docs


def tokenize_text(text):
    return set(re.findall(r"[a-zA-Z0-9+#.]+", text.lower()))


def check_routing_intent(query):
    query_tokens = tokenize_text(query)
    project_keywords = {
        "project", "projects", "portfolio", "practical", "practicals",
        "work", "assignment", "assignments", "built", "developed",
    }
    experience_keywords = {
        "experience", "experiences", "professional", "history", "career",
        "employment", "job", "jobs", "workplace", "company", "companies", "role",
    }
    if query_tokens & project_keywords:
        return "project"
    elif query_tokens & experience_keywords:
        return "experience"
    return "general"


def create_lexical_retriever(splits, k=5):
    indexed_docs = [(doc, tokenize_text(doc.page_content)) for doc in splits]

    def retrieve(query):
        query_tokens = tokenize_text(query)
        if not query_tokens:
            return splits[:k]
        scored = [(len(query_tokens & dt), doc) for doc, dt in indexed_docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        matches = [doc for score, doc in scored if score > 0]
        return (matches or splits)[:k]

    return RunnableLambda(retrieve)


# ─── RAG Initialisation ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Setting up AskSai (one-time)...")
def initialize_rag():
    resume_path = get_resume_path()
    if resume_path is None:
        return None, "No .txt resume/profile file was found in the data folder."

    hf_token = get_config("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        return None, (
            "HUGGINGFACEHUB_API_TOKEN is missing. "
            "Add it to your Streamlit secrets."
        )

    with open(resume_path, "r", encoding="utf-8") as f:
        resume_text = f.read()

    experience_chunks, is_exp_section = [], False
    for line in resume_text.split("\n"):
        if any(kw in line.lower() for kw in ("experience", "employment", "career")):
            is_exp_section = True
        elif any(kw in line.lower() for kw in ("project", "portfolio")):
            is_exp_section = False
        if is_exp_section and line.strip():
            experience_chunks.append(line)

    experience_text = "\n".join(experience_chunks) if experience_chunks else resume_text

    docs = [Document(page_content=resume_text, metadata={"source": resume_path})]
    project_docs = extract_project_docs(resume_text, resume_path)
    experience_docs = [
        Document(
            page_content=experience_text,
            metadata={"source": resume_path, "section": "experience"},
        )
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    general_splits = text_splitter.split_documents(docs)
    retrieval_docs = project_docs + general_splits

    try:
        embedding_model = get_config("HF_EMBEDDING_MODEL_ID", DEFAULT_EMBEDDING_MODEL_ID)
        use_local = get_config("USE_LOCAL_EMBEDDINGS", "false").lower() in {"1", "true", "yes"}

        if use_local:
            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        else:
            embeddings = HuggingFaceEndpointEmbeddings(
                model=embedding_model,
                task="feature-extraction",
                huggingfacehub_api_token=hf_token,
            )
        vectorstore = Chroma.from_documents(documents=retrieval_docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    except Exception as exc:
        st.warning(f"Semantic embeddings unavailable — using keyword retrieval. Details: {exc}")
        retriever = create_lexical_retriever(retrieval_docs, k=8)

    def retrieve_context(query):
        user_input = query["input"]
        intent = check_routing_intent(user_input)
        if intent == "project" and project_docs:
            return format_docs(project_docs)
        if intent == "experience" and experience_docs:
            return format_docs(experience_docs)
        return format_docs(retriever.invoke(user_input))

    model_id = get_config("HF_MODEL_ID", DEFAULT_MODEL_ID)
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id,
        task="conversational",
        temperature=0.1,
        max_new_tokens=1024,
        huggingfacehub_api_token=hf_token,
    )
    llm = ChatHuggingFace(llm=endpoint)

    system_prompt = (
        "You are AskSai, a professional AI assistant representing Sai.\n"
        "Answer questions using ONLY the provided context below.\n"
        "Always refer to the person as 'Sai' — never use 'the candidate', 'he/she', or 'they'.\n"
        "Use natural, confident language such as:\n"
        "  - 'Sai has experience in...'\n"
        "  - 'Sai built this project using...'\n"
        "  - 'Sai's background includes...'\n"
        "  - 'In this role, Sai was responsible for...'\n"
        "If the question is outside the scope of the provided context, or is unrelated to Sai's "
        "professional profile, respond with EXACTLY this message and nothing else:\n"
        f"  '{OUT_OF_SCOPE_REPLY}'\n"
        "Keep responses concise, polite, and factual.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    rag_chain = (
        {
            "context": retrieve_context,
            "input": lambda query: query["input"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, None


# ─── Boot ─────────────────────────────────────────────────────────────────────
rag_chain, rag_error = initialize_rag()
if rag_error:
    st.error(f"❌ Setup Error: {rag_error}")
    st.stop()

# ✅ FIX: Show any sheet logging errors that survived a rerun.
# st.rerun() destroys st.warning() before the user sees it,
# so we store errors in session_state and display them after rerun.
if st.session_state.get("_sheet_log_error"):
    st.warning(f"📋 Sheet logging failed: {st.session_state['_sheet_log_error']}")
    del st.session_state["_sheet_log_error"]

# ─── Chat History ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hi there! I'm **AskSai** — your direct line to Sai's professional profile.\n\n"
                "Ask me about Sai's skills, projects, experience, or tech stack!"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── Input & Response ─────────────────────────────────────────────────────────
if user_query := st.chat_input("Ask anything about Sai..."):
    st.markdown(
        """
        <style>
            div[data-testid="stChatInput"]::before {
                content: "💭 \\A 👽";
                white-space: pre-wrap;
                font-size: 26px;
                position: absolute;
                top: -65px;
                left: 45%;
                text-align: center;
                line-height: 1.2;
                animation: fastBubble 1s ease-in-out infinite alternate;
            }
            @keyframes fastBubble {
                0% { opacity: 0.3; transform: translateY(0px) scale(0.95); }
                100% { opacity: 1; transform: translateY(-4px) scale(1.05); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Looking up Sai's profile..."):
            try:
                answer = rag_chain.invoke({"input": user_query})
            except Exception as e:
                answer = f"⚠️ Something went wrong: {str(e)}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # ── Out-of-scope detection + logging ──
        lowered = answer.lower()
        matched_phrase = next((p for p in OUT_OF_SCOPE_PHRASES if p in lowered), None)
        if matched_phrase:
            log_err = log_out_of_scope(user_query)
            if log_err:
                # ✅ FIX: Store error in session_state so it survives st.rerun()
                # and is visible to the developer after the page reloads.
                st.session_state["_sheet_log_error"] = log_err

    st.rerun()

# ─── Idle alien animation ─────────────────────────────────────────────────────
else:
    st.markdown(
        """
        <style>
            div[data-testid="stChatInput"]::before {
                content: "👽";
                font-size: 28px;
                position: absolute;
                top: -42px;
                left: 20px;
                animation: smoothWalk 6s ease-in-out infinite alternate;
                pointer-events: none;
            }
            @keyframes smoothWalk {
                0%   { left: 5%;  transform: rotate(0deg); }
                25%  { transform: translateY(-3px) rotate(4deg); }
                50%  { transform: translateY(0px)  rotate(-4deg); }
                75%  { transform: translateY(-3px) rotate(4deg); }
                100% { left: 85%; transform: rotate(0deg); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
