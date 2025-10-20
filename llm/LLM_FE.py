import sys, os
from typing import Any, Dict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import streamlit as st
import os
import shutil
import threading
import queue
import time

# Import your custom modules
from chat_handler import build_messages
from LLM_API import call_gemini_with_timing
from LLM_LOCAL import call_lmstudio_with_timing
from config_loader import ui_default_backend, paths_data_dir
from pipeline.pipeline_qa import fetch_retrieval
from pipeline.document_processor import DocumentProcessor
from embedders.providers.ollama import OllamaModelType

# === NEW: ROBUST INITIALIZATION FUNCTION ===
def get_doc_processor():
    """
    Safely initializes and returns the DocumentProcessor instance from session_state.
    This "get-or-create" pattern prevents KeyErrors during unexpected reruns.
    """
    if "doc_processor" not in st.session_state:
        st.session_state["doc_processor"] = DocumentProcessor(
            data_dir=paths_data_dir(),
            model_type=OllamaModelType.GEMMA
        )
    return st.session_state["doc_processor"]

# === REFACTORED: BACKGROUND PROCESSING FUNCTION ===
def process_documents_in_background():
    """
    Handles the document processing workflow in a background thread.
    """
    import time
    
    progress_container = st.empty()
    progress_queue = queue.Queue()
    result_queue = queue.Queue()

    def update_progress_callback(current_file, file_idx, total_files, current_chunk, total_chunks, elapsed):
        progress_queue.put({
            "current_file": current_file,
            "file_idx": file_idx,
            "total_files": total_files,
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
            "elapsed": elapsed
        })

    def processing_thread_target():
        """Target function for the background thread."""
        processor = get_doc_processor()
        result = processor.process_documents(progress_callback=update_progress_callback)
        result_queue.put(result)

    thread = threading.Thread(target=processing_thread_target)
    thread.start()

    # Main thread: Update UI with progress from the queue
    while thread.is_alive() or not progress_queue.empty():
        try:
            progress_data = progress_queue.get(timeout=0.1)
            st.session_state["processing_progress"] = progress_data
            
            # Calculate chunk progress percentage
            current_chunk = progress_data['current_chunk']
            total_chunks = progress_data['total_chunks']
            progress_pct = (current_chunk / total_chunks * 100) if total_chunks > 0 else 0
            progress_value = (current_chunk / total_chunks) if total_chunks > 0 else 0
            
            # Build progress display
            progress_html = f"""
            <div style="margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="font-weight: bold;">{progress_data['current_file']}</span>
                    <span style="font-weight: bold;">{progress_pct:.0f}%</span>
                </div>
            </div>
            """
            
            with progress_container.container():
                st.markdown(progress_html, unsafe_allow_html=True)
                st.progress(progress_value)
                st.markdown(f"Processed files: **{progress_data['file_idx']-1}/{progress_data['total_files']}**")
                st.markdown(f"Total time taken to process **{progress_data['file_idx']-1}** file(s): **{progress_data['elapsed']:.1f}s**")
                
        except queue.Empty:
            pass

    thread.join()
    result = result_queue.get()

    # Finalize state - KEEP display visible, just mark as complete
    total_time = result.get("total_time", 0)
    total_files = result.get("total_files", 0)
    processed = result.get("processed", 0)
    
    # Build final display
    final_html = f"""
    <div style="margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-weight: bold;">✅ All documents processed</span>
            <span style="font-weight: bold;">100%</span>
        </div>
    </div>
    """
    
    with progress_container.container():
        st.markdown(final_html, unsafe_allow_html=True)
        st.progress(1.0)
        st.markdown(f"Processed files: **{processed}/{total_files}**")
        st.markdown(f"Total time taken to process **{processed}** file(s): **{total_time:.1f}s**")
    
    # Wait 10 seconds before clearing
    time.sleep(10)
    
    # Clear the display
    progress_container.empty()
    
    # Mark as NOT processing
    st.session_state["is_processing"] = False
    st.session_state["processing_complete"] = True

# === PAGE CONFIG & STYLES ===
st.set_page_config(page_title="AI Chatbot", page_icon=":speech_balloon:", layout="wide")
css_path = Path(__file__).with_name("chat_styles.css")
st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


# === SIDEBAR ===
with st.sidebar:
    st.markdown("### Menu")
    if st.button("New Chat"):
        st.session_state["messages"] = []
        st.rerun()

    st.button("Recent Chats")
    st.button("Rephrase text...")
    st.button("Fix this code...")
    st.button("Sample Copy for...")
    st.markdown("---")

    # === UPLOAD FILE ===
    st.markdown("### Upload file")
    uploaded_file = st.file_uploader("Chọn file để tải lên", type=["pdf", "docx", "txt"])
    if uploaded_file is not None:
        save_dir_path = f"{paths_data_dir()} + /pdf"
        os.makedirs(save_dir_path, exist_ok=True)
        save_path = os.path.join(str(save_dir_path), uploaded_file.name)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(uploaded_file, f)
        st.success(f"Đã lưu file: {uploaded_file.name}")
    st.markdown("---")

    # === DOCUMENT PROCESSING (using the safe getter) ===
    st.markdown("### Process Documents")

    # Safely get the processor and its status
    doc_processor = get_doc_processor()
    unprocessed = doc_processor.get_unprocessed_pdfs()
    unprocessed_count = len(unprocessed)

    # Initialize UI state flags
    if "is_processing" not in st.session_state:
        st.session_state["is_processing"] = False
    if "processing_progress" not in st.session_state:
        st.session_state["processing_progress"] = {}

    process_disabled = st.session_state["is_processing"] or unprocessed_count == 0
    if st.button(
            f"Process Documents ({unprocessed_count})",
            disabled=process_disabled,
            help="Process unprocessed PDF documents"
        ):
            # RESET all processing states when button clicked
            st.session_state["is_processing"] = True
            st.session_state["processing_progress"] = {
                "current": 0, "total": unprocessed_count, "elapsed": 0,
                "eta": None, "current_file": ""
            }
            # Clear completion flag to reset display
            st.session_state["processing_complete"] = False
            
            process_documents_in_background()
            st.rerun()

    if not st.session_state["is_processing"] and unprocessed_count == 0:
        st.info("All documents processed")
    st.markdown("---")

    # === BACKEND SELECTION ===
    st.markdown("<div style='flex: 1;'></div>", unsafe_allow_html=True)
    backend_options = ["gemini", "lmstudio"]
    if "backend_mode" not in st.session_state:
        default_backend = ui_default_backend()
        st.session_state["backend_mode"] = default_backend if default_backend in backend_options else backend_options[0]

    st.markdown("<div class='sidebar-footer'>", unsafe_allow_html=True)
    st.radio(
        "Response source",
        backend_options,
        key="backend_mode",
        format_func=lambda x: "Gemini API" if x == "gemini" else "LM Studio Local"
    )
    st.markdown("Welcome back", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

backend = st.session_state["backend_mode"]

# === SESSION STATE INIT for Chat ===
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []


# === MAIN CHAT INTERFACE ===
st.markdown("<div class='chat-header'>Chat Window</div>", unsafe_allow_html=True)

# Render chat history
chat_html_parts = ["<div class='chat-log'>"]
for msg in st.session_state["messages"]:
    role = "bot" if msg.get("role") == "assistant" else "user"
    bubble = f"<div class='chat-row {role}'><div class='chat-bubble {role}'>{msg.get('content', '')}</div>"
    if msg.get("role") == "assistant":
        if msg.get("time_taken"):
            bubble += f"<div class='time-info'>⏱️ Time taken: {msg['time_taken']:.2f}s</div>"
        # Add token usage info
        if msg.get("total_tokens", 0) > 0:
            bubble += f"<div class='time-info'>🔢 Tokens: {msg.get('prompt_tokens', 0)} prompt + {msg.get('response_tokens', 0)} response = {msg.get('total_tokens', 0)} total</div>"
    bubble += "</div>"
    chat_html_parts.append(bubble)
if st.session_state.get("is_generating"):
    chat_html_parts.append("<div class='chat-row bot'><div class='chat-bubble bot'><span class='typing'><span></span><span></span><span></span></span></div></div>")
chat_html_parts.append("</div>")
st.markdown("".join(chat_html_parts), unsafe_allow_html=True)

# === RETRIEVAL SOURCES (UI) ===
# === RETRIEVAL SOURCES (UI) ===
sources = st.session_state.get("last_sources", [])
if sources:
    st.markdown("### Nguồn tham khảo")
    for i, src in enumerate(sources, 1):
        file_name = src.get("file_name", "?")
        
        # Fix: Try multiple keys for page number
        page = src.get("page_number")
        if page is None:
            # Fallback to page_numbers list if page_number is None
            page_numbers = src.get("page_numbers", [])
            if page_numbers:
                page = page_numbers[0]  # Get first page from list
            else:
                page = "?"
        
        try:
            score = float(src.get("similarity_score", 0.0))
        except Exception:
            score = 0.0
        text = src.get("snippet", "") or ""
        snippet = text
        
        st.markdown(f"- [{i}] {file_name} - trang {page} (điểm {score:.3f})")
        with st.expander(f"Xem trích đoạn {i}"):
            if snippet.strip():
                st.markdown(snippet)
            else:
                st.write("Không có nội dung trích đoạn")
else:
    st.info("Chưa có nguồn tham khảo nào được tìm thấy. Hãy đặt câu hỏi để hệ thống tìm kiếm tài liệu liên quan.")

# === BACKEND CALL ===
def ask_backend(prompt_text: str) -> Dict[str, Any]:
    """
    Xử lý request tới LLM backend
    
    Args:
        prompt_text: User query
    
    Returns:
        Response từ LLM
    """
    try:
        # TODO: Khi có retrieval system, lấy context ở đây
        context = ""  # Tạm thời để trống
        
        # Build messages bằng chat_handler
        # Lấy context từ Retrieval (nếu có) và lưu nguồn để hiển thị.
        try:
            ret = fetch_retrieval(prompt_text, top_k=10, max_chars=8000)  # Tăng lên 8000
            context = ret.get("context", "") or ""
            st.session_state["last_sources"] = ret.get("sources", [])
        except Exception:
            context = ""
            st.session_state["last_sources"] = []

        messages = build_messages(
            query=prompt_text,
            context=context,
            history=st.session_state["messages"]
        )
        
        # Gọi LLM tương ứng
        if backend == "gemini":
            reply = call_gemini_with_timing(messages)
        else:  # lmstudio
            reply = call_lmstudio_with_timing(messages)
        
        return reply
    
    except Exception as e:
        # Return dict format to match API functions
        return {
            "response": f"[Error] {e}",
            "time_taken": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0
        }

# === CHAT INPUT & RESPONSE GENERATION ===
if prompt := st.chat_input("Type a new message here", disabled=st.session_state["is_generating"]):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.session_state["pending_prompt"] = prompt
    st.session_state["is_generating"] = True
    st.rerun()

if st.session_state["is_generating"] and st.session_state["pending_prompt"]:
    start_time = time.time()
    result = ask_backend(st.session_state["pending_prompt"])
    total_time = time.time() - start_time  # Calculate total time including retrieval
    with st.spinner("Assistant is typing..."):
        result = ask_backend(st.session_state["pending_prompt"])
    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["response"],
        "time_taken": total_time,  # Use total time instead of result["time_taken"]
        "prompt_tokens": result.get("prompt_tokens", 0),
        "response_tokens": result.get("response_tokens", 0),
        "total_tokens": result.get("total_tokens", 0)
    })
    st.session_state["pending_prompt"] = None
    st.session_state["is_generating"] = False
    st.rerun()