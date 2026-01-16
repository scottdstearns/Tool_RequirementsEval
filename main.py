import os
import streamlit as st
import pandas as pd
import json
import tempfile
from datetime import datetime
import io
import platform
from pathlib import Path
import numpy as np
from typing import List, Dict

from app.utils.document_parser import RequirementParser
from app.utils.retriever import retrieve_topk, join_context
from app.utils.requirements_evaluator import RequirementsEvaluator
from app.models.requirement import Requirement, RequirementEvaluation
from app.evaluation_criteria import INCOSE_CRITERIA, SIMPLIFIED_CRITERIA
from app.settings import Settings
from qdrant_client import QdrantClient

# Initialize settings and validate configuration
settings = Settings()

def test_litellm_connectivity():
    """Test connectivity to LiteLLM proxy on startup"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
        # Simple test with minimal tokens to check connectivity
        response = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=5,
            timeout=10
        )
        print("✅ LiteLLM connectivity test: SUCCESS")
        return True
    except Exception as e:
        print(f"⚠️  LiteLLM connectivity test: FAILED - {str(e)}")
        print("   App will continue but may encounter issues during evaluation.")
        return False

# Test connectivity at startup (non-blocking)
try:
    test_litellm_connectivity()
except:
    pass  # Don't block app startup if test fails

# Info about knowledge base
def df_to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
    buf.seek(0)
    return buf.getvalue()

# Platform detection
IS_WINDOWS = platform.system() == "Windows"

# Helper function for cross-platform path handling
def normalize_path(path_str):
    """Normalize file path for current operating system"""
    return str(Path(path_str))

# Page configuration
st.set_page_config(
    page_title="Requirements Evaluation Tool",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'requirements' not in st.session_state:
    st.session_state.requirements = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = "idle"  # idle, processing, completed, error
if 'evaluation_status' not in st.session_state:
    st.session_state.evaluation_status = "idle"  # idle, evaluating, completed, error
if 'status_message' not in st.session_state:
    st.session_state.status_message = ""

def process_uploaded_file(uploaded_file):
    """Process the uploaded requirements file"""
    if uploaded_file is None:
        st.session_state.status_message = "No file selected. Please upload a requirements document."
        st.session_state.processing_status = "error"
        return False
    
    # Update processing status
    print(f"Processing {uploaded_file.name}...")
    st.session_state.processing_status = "processing"
    st.session_state.status_message = f"Processing {uploaded_file.name}... Please wait."
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_path = normalize_path(os.path.join(temp_dir, uploaded_file.name))
    
    # Write the uploaded file to the temporary location
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Parse the requirements from the file
    print(f"Parsing requirements from {temp_path}...")
    parser = RequirementParser()
    print(f"Supported file: {parser.is_supported_file(temp_path)}")
    
    if not parser.is_supported_file(temp_path):
        st.session_state.status_message = f"Unsupported file format: {uploaded_file.name}. Please upload a .docx, .pdf, .xlsx, or .csv file."
        st.session_state.processing_status = "error"
        return False
    
    try:
        print(f"Extracting requirements from {temp_path}...")
        requirements_data = parser.extract_requirements(temp_path)
        print(f"Extracted {len(requirements_data)} requirements")
        
        if not requirements_data:
            st.session_state.status_message = f"No requirements found in {uploaded_file.name}. Please check the file format and try again."
            st.session_state.processing_status = "error"
            return False
        
        # Convert to Requirement objects
        st.session_state.requirements = [
            Requirement.from_dict(req_dict, uploaded_file.name)
            for req_dict in requirements_data
        ]
        
        st.session_state.processed_file = uploaded_file.name
        st.session_state.processing_status = "completed"
        st.session_state.status_message = f"Successfully processed {len(st.session_state.requirements)} requirements from {uploaded_file.name}"
        return True
        
    except Exception as e:
        st.session_state.status_message = f"Error parsing the requirements: {str(e)}"
        st.session_state.processing_status = "error"
        return False

def evaluate_requirements():
    """Evaluate the requirements using the selected criteria"""
    if not st.session_state.requirements:
        st.session_state.status_message = "No requirements to evaluate. Please upload and process a requirements document first."
        st.session_state.evaluation_status = "error"
        return
    
    st.session_state.evaluation_status = "evaluating"
    
    criteria = SIMPLIFIED_CRITERIA if st.session_state.criteria_option == "Simplified" else INCOSE_CRITERIA
    st.session_state.status_message = f"Evaluating requirements using {criteria}... This may take a few minutes depending on the number of requirements."

    # ----------------------------
    # build a retriever function if "Use KB" is on
    # ----------------------------
    kb_retriever_fn = None
    if st.session_state.get("use_kb", False):
        def kb_retriever_fn(text: str) -> str:
            chunks = retrieve_topk(text, top_k=int(st.session_state.get("top_k", 4)))
            return join_context(chunks)  # plain text block we’ll prepend to the prompt

    
    try:
        with st.spinner("Evaluating requirements... Please do not close the browser window."):
            # hand the triever into the evaluator if selected
            #evaluator = RequirementsEvaluator(criteria=criteria)
            evaluator = RequirementsEvaluator(criteria=criteria, retriever=kb_retriever_fn)
            
            # Process requirements in smaller batches with retry logic
            #evaluated_requirements = []
            evaluated_requirements: list[tuple[Requirement, RequirementEvaluation]] = []
            batch_size = 5  # Process 5 requirements at a time
            max_retries = 3
            
            for i in range(0, len(st.session_state.requirements), batch_size):
                batch = st.session_state.requirements[i:i + batch_size]
                retry_count = 0
                
                while retry_count < max_retries:
                    try:
                        batch_results = evaluator.evaluate_requirements(batch)
                        evaluated_requirements.extend(batch_results)
                        st.session_state.status_message = f"Processed {len(evaluated_requirements)} of {len(st.session_state.requirements)} requirements..."
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count == max_retries:
                            raise Exception(f"Failed to evaluate requirements after {max_retries} retries. Last error: {str(e)}")
                        st.warning(f"Retrying batch {i//batch_size + 1}... (Attempt {retry_count + 1}/{max_retries})")
                        import time
                        time.sleep(2)  # Wait 2 seconds before retrying
            
            # Generate report
            '''# --- Generate report robustly (ID/Text first, no duplicate-column crashes) ---

            # Evaluator returns a DataFrame. Depending on your implementation, it may
            # already contain "ID" and "Text" (if it consumes pairs), or it may not.
            report_df = evaluator.generate_evaluation_report(evaluated_requirements)

            # 0) Normalize and drop duplicate headers (keep first)
            report_df.columns = [str(c).strip() for c in report_df.columns]
            report_df = report_df.loc[:, ~pd.Index(report_df.columns).duplicated(keep="first")]

            # 1) If ID/Text are missing, build them from the original inputs and prepend
            if not {"ID", "Text"}.issubset(report_df.columns):
                meta_df = pd.DataFrame([{"ID": r.id, "Text": r.text} for r in st.session_state.requirements])
                meta_df = meta_df.reset_index(drop=True)
                report_df = report_df.reset_index(drop=True)
                report_df = pd.concat([meta_df, report_df], axis=1)
                # re-normalize in case concat introduced dupes
                report_df.columns = [str(c).strip() for c in report_df.columns]
                report_df = report_df.loc[:, ~pd.Index(report_df.columns).duplicated(keep="first")]
            
            # 2) Deterministic ordering:
            base = ["ID", "Text", "Overall Score", "Status", "Suggested Rewrite"]
            
            # collect criterion base names from existing columns
            crit_bases = []
            for c in report_df.columns:
                if c.endswith("(score)"):
                    crit_bases.append(c[:-8].strip())
                elif c.endswith("(explanation)"):
                    crit_bases.append(c[:-14].strip())
            
            # preserve first-seen order
            seen = set()
            crit_bases = [x for x in crit_bases if not (x in seen or seen.add(x))]
            
            ordered = [c for c in base if c in report_df.columns]
            for b in crit_bases:
                s, e = f"{b} (score)", f"{b} (explanation)"
                if s in report_df.columns: ordered.append(s)
                if e in report_df.columns: ordered.append(e)
            
            # keep any not-yet-accounted columns at the end (avoid reindex crashes)
            remaining = [c for c in report_df.columns if c not in ordered]
            final_df = report_df.loc[:, ordered + remaining]
            
            st.session_state.evaluation_results = final_df
            st.session_state.evaluation_status = "completed"
            st.session_state.status_message = f"Evaluation completed successfully for {len(evaluated_requirements)} requirements."
            ###
            '''
            # --- Generate report robustly: ID/Text first, no duplicate-column crashes ---
            assert evaluated_requirements and \
                   isinstance(evaluated_requirements[0][0], Requirement) and \
                   isinstance(evaluated_requirements[0][1], RequirementEvaluation)
            
            report_df = evaluator.generate_evaluation_report(evaluated_requirements)
            
            # 0) Normalize headers + drop duplicates (keep first)
            report_df.columns = [str(c).strip() for c in report_df.columns]
            report_df = report_df.loc[:, ~pd.Index(report_df.columns).duplicated(keep="first")]
            
            # 1) Standardize column name to "Text" (your report uses "Requirement Text")
            if "Requirement Text" in report_df.columns and "Text" not in report_df.columns:
                report_df = report_df.rename(columns={"Requirement Text": "Text"})
            
            # 2) Deterministic order:
            base = ["ID", "Text", "Overall Score", "Status", "Suggested Rewrite"]
            
            # Collect criterion base names from existing columns
            crit_bases = []
            for col in report_df.columns:
                if col.endswith("(score)"):
                    crit_bases.append(col[:-8].strip())
                elif col.endswith("(explanation)"):
                    crit_bases.append(col[:-14].strip())
            
            # Preserve first-seen order
            seen = set()
            crit_bases = [c for c in crit_bases if not (c in seen or seen.add(c))]
            
            ordered = [c for c in base if c in report_df.columns]
            for b in crit_bases:
                s, e = f"{b} (score)", f"{b} (explanation)"
                if s in report_df.columns: ordered.append(s)
                if e in report_df.columns: ordered.append(e)
            
            # Keep any remaining columns at the end (avoid reindex with dupes)
            remaining = [c for c in report_df.columns if c not in ordered]
            final_df = report_df.loc[:, ordered + remaining]
            
            st.session_state.evaluation_results = final_df
            st.session_state.evaluation_status = "completed"
            st.session_state.status_message = f"Evaluation completed successfully for {len(final_df)} requirements."
            
    except Exception as e:
        st.session_state.status_message = f"Error during evaluation: {str(e)}\nPlease try again or contact support if the issue persists."
        st.session_state.evaluation_status = "error"

def download_report():
    """Generate downloadable report"""
    if st.session_state.evaluation_results is None:
        st.error("No evaluation results to download.")
        return
    
    # Create timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Excel report
    try:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.evaluation_results.to_excel(writer, index=False, sheet_name='Requirements Evaluation')
        
        st.download_button(
            label="Download Excel Report",
            data=buffer.getvalue(),
            file_name=f"requirements_evaluation_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"Error generating Excel report: {str(e)}")
    
    # CSV report
    try:
        csv_data = st.session_state.evaluation_results.to_csv(index=False)
        st.download_button(
            label="Download CSV Report",
            data=csv_data,
            file_name=f"requirements_evaluation_{timestamp}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"Error generating CSV report: {str(e)}")

# Main app interface
st.title("Requirements Evaluation Tool")
st.markdown("""
This tool helps systems engineers evaluate and improve their requirements based on industry standards.
Upload a requirements document to get started.
""")

# Status indicator
if st.session_state.status_message:
    if "error" in st.session_state.processing_status or "error" in st.session_state.evaluation_status:
        st.error(st.session_state.status_message)
    elif "completed" in st.session_state.processing_status or "completed" in st.session_state.evaluation_status:
        st.success(st.session_state.status_message)
    else:
        st.info(st.session_state.status_message)

# Progress indicator
col1, col2 = st.columns(2)
with col1:
    st.subheader("Processing Status")
    if st.session_state.processing_status == "idle":
        st.info("Waiting for file upload")
    elif st.session_state.processing_status == "processing":
        st.warning("Processing file...")
    elif st.session_state.processing_status == "completed":
        st.success("Processing completed ✓")
    else:
        st.error("Processing error ✗")

with col2:
    st.subheader("Evaluation Status")
    if st.session_state.evaluation_status == "idle":
        st.info("Waiting for evaluation request")
    elif st.session_state.evaluation_status == "evaluating":
        st.warning("Evaluating requirements...")
    elif st.session_state.evaluation_status == "completed":
        st.success("Evaluation completed ✓")
    else:
        st.error("Evaluation error ✗")

# Sidebar for file upload and options
with st.sidebar:
    st.header("Upload Requirements")
    uploaded_file = st.file_uploader("Choose a requirements file", 
                                    type=["docx", "pdf", "xlsx", "csv"],
                                    help="Upload a document containing requirements.")
    
    if uploaded_file is not None:
        process_button = st.button("Process Requirements")
        if process_button:
            success = process_uploaded_file(uploaded_file)
            if not success:
                st.error("Failed to process requirements. See status message for details.")
    
    # side bar feature - option to use knowledge base
    use_kb = st.sidebar.checkbox("Use Knowledge Base context (Qdrant)", value=True)
    top_k = st.sidebar.slider("Top-K context chunks", 1, 8, 4)

    # persist in session_state so the worker function can see them
    st.session_state.use_kb = use_kb
    st.session_state.top_k = top_k

      # --- KB health badge ---
    try:
        c = QdrantClient(url=os.getenv("QDRANT_URL","http://qdrant:6333"),
                         api_key=os.getenv("QDRANT_API_KEY"), prefer_grpc=False)
        cols = [x.name for x in c.get_collections().collections]
        kb_col = os.getenv("KB_COLLECTION","(unset)")
        kb_name = os.getenv("KB_NAME","(none)")
        ok = "✅" if kb_col in cols else "⚠️"
        st.caption(f"{ok} KB: `{kb_col}` • Filter: `{kb_name}`")
    except Exception as e:
        st.caption(f"⚠️ KB check failed: {e}")


    st.header("Evaluation Options")
    criteria_options = ["Simplified", "Full INCOSE"]
    st.session_state.criteria_option = st.radio("Evaluation Criteria", criteria_options)
    
    evaluate_button_disabled = len(st.session_state.requirements) == 0
    
    if evaluate_button_disabled:
        st.warning("Process requirements first before evaluation")
    
    evaluate_button = st.button("Evaluate Requirements", disabled=evaluate_button_disabled)
    if evaluate_button:
        evaluate_requirements()

# Main content area
if st.session_state.processing_status == "idle":
    st.info("Please upload a requirements document using the sidebar and click 'Process Requirements'.")
elif st.session_state.requirements and st.session_state.processing_status == "completed":
    st.subheader(f"Requirements from {st.session_state.processed_file}")
    st.write(f"Found {len(st.session_state.requirements)} requirements.")
    
    # Display requirements table
    print("\nDebugging requirements data:")
    print(f"Number of requirements: {len(st.session_state.requirements)}")
    
    # Inspect each requirement
    for i, req in enumerate(st.session_state.requirements):
        print(f"\nRequirement {i}:")
        print(f"Type: {type(req)}")
        print(f"ID type: {type(req.id)}")
        print(f"ID value: {req.id}")
        print(f"Text type: {type(req.text)}")
        print(f"Text value: {req.text}")
    
    # Create DataFrame using from_records
    data = []
    for req in st.session_state.requirements:
        data.append({
            "ID": str(req.id) if req.id is not None else "",
            "Requirement Text": str(req.text) if req.text is not None else ""
        })
    
    print(f"Data: {data}")
    #########################################################
   # import pandas as pd
    #import numpy as np # Import numpy if values might be numpy types

    # Your code that leads up to defining Data...
    # Data = ...? Assume it's defined somewhere above

    print("--- Debugging ---")
    print(f"Value of Data:\n{data}")
    print(f"Type of Data: {type(data)}")

    # Check the type of the first element if Data is a list/tuple
    if isinstance(data, (list, tuple)) and len(data) > 0:
        print(f"Type of first element in Data: {type(data[0])}")
        # If the first element is a dictionary, check the types of its values
        if isinstance(data[0], dict):
            print("Types of values in the first dictionary:")
            for key, value in data[0].items():
                print(f"  Key '{key}': Type {type(value)}")

    print("--- End Debugging ---")

    # Attempt to create the DataFrame (this is the line that might fail)
    try:
        df = pd.DataFrame(data)
        print("\nDataFrame created successfully:")
        print(df)
    except TypeError as e:
        print(f"\nCaught TypeError during DataFrame creation: {e}")
        print("Check the debug output above to see the structure and types within 'Data'.")
    except Exception as e:
        print(f"\nCaught other exception during DataFrame creation: {e}")

    #########################################################
    # Create DataFrame using from_records
    req_df = pd.DataFrame.from_records(data)
    
    print(f"Req_df: {req_df}")
    st.dataframe(req_df, use_container_width=True)
    
    if st.session_state.evaluation_status == "idle":
        st.info("Click 'Evaluate Requirements' in the sidebar to analyze these requirements.")

# Display evaluation results if available
if st.session_state.evaluation_results is not None and st.session_state.evaluation_status == "completed":
    st.subheader("Evaluation Results")
    
    # Summary statistics
    results_df = st.session_state.evaluation_results
    total_reqs = len(results_df)
    passed_reqs = len(results_df[results_df['Status'] == 'Pass'])
    needs_improvement = total_reqs - passed_reqs
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Requirements", total_reqs)
    col2.metric("Passed", passed_reqs)
    col3.metric("Needs Improvement", needs_improvement)
    
    # Display full results with color coding
    st.dataframe(
        results_df.style.apply(
            lambda x: ['background-color: #d4f1dd' if x['Status'] == 'Pass' else 'background-color: #f7d1d1' for _ in x],
            axis=1
        ),
        use_container_width=True
    )
    
    # Download options
    st.subheader("Download Report")
    download_report()

# Footer
st.markdown("---")
st.markdown("© 2025 Requirements Evaluation Tool | Powered by Azure OpenAI")

# Run the app with: streamlit run app.py
