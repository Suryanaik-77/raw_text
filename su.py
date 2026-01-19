import streamlit as st

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Document Processing Workflow",
    layout="centered"
)

# -----------------------------
# Step State
# -----------------------------
if "step" not in st.session_state:
    st.session_state.step = 0

TOTAL_STEPS = 6

# -----------------------------
# Dark Theme Styling
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #0e0e0e;
}
.block-container {
    padding-top: 2rem;
}

.step-card {
    background: #151515;
    padding: 32px;
    border-radius: 18px;
    border-left: 6px solid #22c55e;
    color: #e5e7eb;
    box-shadow: 0 0 30px rgba(34,197,94,0.25);
    margin-bottom: 25px;
}

.step-title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 12px;
}

.step-desc {
    font-size: 16px;
    color: #9ca3af;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 14px;
    margin-right: 10px;
}

.badge-on {
    background: #22c55e;
    color: black;
}

.badge-off {
    background: transparent;
    border: 1px solid #6b7280;
    color: #9ca3af;
}

.next-btn button {
    background: linear-gradient(90deg,#22c55e,#16a34a);
    color: black;
    border-radius: 999px;
    padding: 10px 28px;
    font-weight: 600;
    font-size: 16px;
    box-shadow: 0 0 15px rgba(34,197,94,0.4);
}

.progress {
    color: #9ca3af;
    margin-bottom: 12px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Progress Indicator
# -----------------------------
st.markdown(
    f"<div class='progress'>Step {st.session_state.step + 1} of {TOTAL_STEPS}</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Step 0 — All Documents
# -----------------------------
if st.session_state.step == 0:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">📁 All Documents</div>
        <div class="step-desc">
            All source documents (PDFs) are collected into a single folder.
            This folder acts as the entry point for the entire document
            processing pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Next Step →", key="next_0"):
        st.session_state.step += 1

# -----------------------------
# Step 1 — PDF Processing
# -----------------------------
elif st.session_state.step == 1:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">📄 PDF Processing</div>
        <div class="step-desc">
            Each PDF is parsed to extract raw text.
            This includes text normalization, encoding fixes,
            and page-wise extraction.
        </div>
        <br>
        <span class="badge badge-on">table_extraction</span>
        <span class="badge badge-off">image_summary</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Next Step →", key="next_1"):
        st.session_state.step += 1

# -----------------------------
# Step 2 — Conditional Extraction
# -----------------------------
elif st.session_state.step == 2:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">🔀 Conditional Extraction</div>
        <div class="step-desc">
            Based on configuration flags:
            <ul>
                <li><b>table_extraction</b> → extract tables</li>
                <li><b>image_summary</b> → summarize images</li>
            </ul>
            These steps are executed only if enabled.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Next Step →", key="next_2"):
        st.session_state.step += 1

# -----------------------------
# Step 3 — Metadata Extraction
# -----------------------------
elif st.session_state.step == 3:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">🧠 Metadata Extraction</div>
        <div class="step-desc">
            The first <b>6000 characters</b> of raw text are sent to the LLM.
            The model extracts:
            <ul>
                <li>Domain</li>
                <li>Stage</li>
                <li>Type</li>
                <li>Tool</li>
                <li>Vendor</li>
                <li>Version</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Next Step →", key="next_3"):
        st.session_state.step += 1

# -----------------------------
# Step 4 — Chunking
# -----------------------------
elif st.session_state.step == 4:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">✂️ Chunking</div>
        <div class="step-desc">
            Raw text is split into smaller,
            semantically meaningful chunks.
            This improves embedding quality
            and retrieval accuracy.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Next Step →", key="next_4"):
        st.session_state.step += 1

# -----------------------------
# Step 5 — Embedding
# -----------------------------
elif st.session_state.step == 5:
    st.markdown("""
    <div class="step-card">
        <div class="step-title">🧬 Embedding & Storage</div>
        <div class="step-desc">
            Text chunks are converted into vector embeddings
            using OpenAI embeddings and stored in Milvus.
            The data is now ready for RAG-based querying.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Workflow Completed Successfully")

    if st.button("🔄 Restart Workflow"):
        st.session_state.step = 0
