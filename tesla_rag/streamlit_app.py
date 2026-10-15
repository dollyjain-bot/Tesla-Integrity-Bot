"""
Tesla Integrity Bot - Streamlit Demonstration App
Simplified Enterprise RAG Application showcasing Mock Login, RAG Chat,
and Retrieved Source Chunks Inspection.
"""

import time
import streamlit as st
from rag_engine import RAGEngine

# ------------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & DARK THEME CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Tesla Integrity Bot - RAG Demo",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0E1117;
        color: #E0E6ED;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    }
    h1, h2, h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .source-badge {
        display: inline-block;
        background-color: #21262D;
        border: 1px solid #30363D;
        color: #58A6FF;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    .chunk-box {
        background-color: #0D1117;
        border-left: 4px solid #E82127;
        border-radius: 6px;
        padding: 14px;
        margin-bottom: 12px;
        font-size: 0.9rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# ------------------------------------------------------------------------------
# 2. CACHED RAG ENGINE INITIALIZATION
# ------------------------------------------------------------------------------

@st.cache_resource
def load_rag_engine():
    """Initializes and caches the RAGEngine instance."""
    return RAGEngine(data_dir="data", vector_dir="./vector_store")


# ------------------------------------------------------------------------------
# 3. SESSION STATE INITIALIZATION
# ------------------------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ------------------------------------------------------------------------------
# 4. SIMPLE MOCK LOGIN PAGE (No Roles/Permissions)
# ------------------------------------------------------------------------------
def render_login_page():
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
            <div style="text-align: center; margin-bottom: 24px;">
                <h1 style="margin-bottom: 4px;">⚡ TESLA INTEGRITY BOT</h1>
                <p style="color: #8B949E; font-size: 1.05rem;">Corporate Ethics & Compliance RAG Demonstration</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            st.subheader("Sign In")
            username_input = st.text_input("Username", value="admin@tesla.demo")
            password_input = st.text_input("Password", type="password", value="demo123")
            submit_button = st.form_submit_button("Login ➔", use_container_width=True)

            if submit_button:
                if username_input.strip():
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username_input.strip()
                    st.success("Authentication successful. Redirecting...")
                    time.sleep(0.3)
                    st.rerun()
                else:
                    st.error("Please enter a username.")


# ------------------------------------------------------------------------------
# 5. MAIN RAG CHAT INTERFACE
# ------------------------------------------------------------------------------
def render_main_app(engine: RAGEngine):
    # Simple Sidebar Controls
    with st.sidebar:
        st.markdown("## ⚡ **RAG Settings**")
        st.markdown("---")

        top_k = st.slider("Top K Chunks", min_value=1, max_value=5, value=3, step=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.1)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, value=0.1, step=0.1)

        st.markdown("---")
        st.write(f"👤 **User:** `{st.session_state['username']}`")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state["authenticated"] = False
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("---")
        try:
            st.caption(f"ChromaDB Chunks: {engine.vector_store.count()}")
        except Exception:
            pass

    # App Title & Subtitle
    st.markdown(
        """
        <div style="border-bottom: 1px solid #30363D; padding-bottom: 12px; margin-bottom: 24px;">
            <h1 style="margin-bottom: 2px;">Tesla Integrity Bot</h1>
            <p style="color: #8B949E; font-size: 1.05rem; margin-top: 0;">
                Corporate Ethics & Compliance RAG Demonstration
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Prompt user to ask a question if no active messages
    if not st.session_state["messages"]:
        st.markdown("### 💬 **Ask a new question:**")
        st.caption("Enter your prompt below to retrieve policy source chunks and generate a grounded response.")

    # Display Chat History
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "response_data" in msg and msg["response_data"]:
                res_data = msg["response_data"]

                # Display Supporting Sources
                if res_data.get("sources"):
                    st.markdown("##### 📌 **Sources**")
                    for src in res_data["sources"]:
                        st.markdown(
                            f"<span class='source-badge'>📄 <b>{src['document_id']}</b> ({src['source']}) | Section: <i>{src['section']}</i></span>",
                            unsafe_allow_html=True,
                        )

                # Display Retrieved Chunks Expander
                if res_data.get("retrieved_chunks"):
                    with st.expander("🔍 Retrieved Source Chunks"):
                        for c_idx, chunk in enumerate(res_data["retrieved_chunks"], start=1):
                            st.markdown(
                                f"""
                                <div class="chunk-box">
                                    <b>Chunk #{c_idx}</b><br>
                                    <b>Source:</b> {chunk.get('source')}<br>
                                    <b>Document ID:</b> {chunk.get('document_id')}<br>
                                    <b>Section:</b> {chunk.get('section')}<br>
                                    <b>Similarity Score:</b> <span style="color:#58A6FF;">{chunk.get('similarity_score', 'N/A')}</span>
                                    <hr style="border: 0.5px solid #30363D; margin: 8px 0;">
                                    <i>"{chunk.get('retrieved_text', chunk.get('chunk_text'))}"</i>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

    # Clear Response Button (displayed after an answer has been generated)
    if st.session_state["messages"]:
        st.markdown("<br>", unsafe_allow_html=True)
        col_clear, col_space = st.columns([1, 4])
        with col_clear:
            if st.button("🔄 Clear Response", use_container_width=True):
                st.session_state["messages"] = []
                st.rerun()

    # Chat Input
    if user_query := st.chat_input("Ask a question about Tesla corporate ethics or compliance..."):
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving policy context & generating response..."):
                try:
                    rag_response = engine.answer_query(
                        query=user_query,
                        top_k=top_k,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    answer_text = rag_response.get("answer", "No answer generated.")
                except Exception as e:
                    rag_response = {}
                    answer_text = f"Error processing query: {str(e)}"

                st.markdown(answer_text)

                if rag_response.get("sources"):
                    st.markdown("##### 📌 **Sources**")
                    for src in rag_response["sources"]:
                        st.markdown(
                            f"<span class='source-badge'>📄 <b>{src['document_id']}</b> ({src['source']}) | Section: <i>{src['section']}</i></span>",
                            unsafe_allow_html=True,
                        )

                if rag_response.get("retrieved_chunks"):
                    with st.expander("🔍 Retrieved Source Chunks"):
                        for c_idx, chunk in enumerate(rag_response["retrieved_chunks"], start=1):
                            st.markdown(
                                f"""
                                <div class="chunk-box">
                                    <b>Chunk #{c_idx}</b><br>
                                    <b>Source:</b> {chunk.get('source')}<br>
                                    <b>Document ID:</b> {chunk.get('document_id')}<br>
                                    <b>Section:</b> {chunk.get('section')}<br>
                                    <b>Similarity Score:</b> <span style="color:#58A6FF;">{chunk.get('similarity_score', 'N/A')}</span>
                                    <hr style="border: 0.5px solid #30363D; margin: 8px 0;">
                                    <i>"{chunk.get('retrieved_text', chunk.get('chunk_text'))}"</i>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": answer_text,
                "response_data": rag_response,
            }
        )
        st.rerun()


# ------------------------------------------------------------------------------
# 6. MAIN ENTRY POINT
# ------------------------------------------------------------------------------
def main():
    engine = load_rag_engine()

    if not st.session_state["authenticated"]:
        render_login_page()
    else:
        render_main_app(engine)


if __name__ == "__main__":
    main()

