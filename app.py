import streamlit as st
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from retriever import retrieve_trials,collection
from reasoner import generate_explanation

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# FIRST streamlit command called
# Sets browser tab title, icon, and layout
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Trial Matcher",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────
# CUSTOM CSSl
# ─────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .trial-card {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .similarity-badge {
        background-color: #28a745;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .warning-text {
        color: #dc3545;
        font-weight: bold;
    }
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 0.5rem;
        font-size: 0.9rem;
        color: #856404;
    }
            
    /* This targets the '500' or count value */
    [data-testid="stMetricValue"] {
        font-size: 24px !important; 
        }

    /* This targets the 'Multiple conditions' or 'Recruiting Only' labels */
    [data-testid="stMetricLabel"] p {
        font-size: 16px !important;
    }  


    </style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HEADER SECTION
# ─────────────────────────────────────────
st.markdown('<p class="main-header">🏥 Clinical Trial Matcher</p>', 
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">Find relevant clinical trials using AI-powered semantic search</p>', 
            unsafe_allow_html=True)

# Medical disclaimer
st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong> This tool is for informational purposes only 
and does not constitute medical advice. Always consult with a qualified 
healthcare professional before participating in any clinical trial.
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────

# Two columns — input on left, settings on right
# st.columns([3,1]) means left column is 3x wider than right column
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📋 Patient Profile")
    
    # Text area for patient description
    patient_query = st.text_area(
        "Describe your medical condition, age, current medications, and relevant history:",
        height=150,
        placeholder="Example: I am a 45-year-old male with Type 2 diabetes. "
                   "My HbA1c is 8.2%. I have been on metformin for 3 years "
                   "but it has stopped being effective. I have no prior insulin use "
                   "and no kidney or liver problems.",
        help="Be as specific as possible — include age, diagnosis, "
             "current medications, lab values, and medical history"
    )

with col2:
    st.subheader("⚙️ Settings")
    
    # Slider for number of results
    # min=1, max=10, default=3, step=1
    n_results = st.slider(
        "Number of trials to find:",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="More results = longer analysis time"
    )
    
    st.markdown("**Model:**")
    st.markdown("`llama-3.3-70b`")
    st.markdown("**Database:**")
    st.markdown(f"`10,000+ Multiple Condition trials`")
    st.markdown("**Embeddings:**")
    st.markdown("`BiomedBERT`")

# ─────────────────────────────────────────
# SEARCH BUTTON
# ─────────────────────────────────────────
st.markdown("---")

# Center the button using columns
_, center, _ = st.columns([2, 1, 2])
with center:
    search_clicked = st.button(
        "🔍 Find Matching Trials",
        type="primary",
        use_container_width=True
    )

# ─────────────────────────────────────────
# RESULTS SECTION
# ─────────────────────────────────────────
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def is_valid_medical_query(query):
    """
    Uses Llama to validate if input is
    a genuine medical query
    Fast — uses smallest/fastest model
    Single yes/no response
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            # Using fast 8B for validation
        
            messages=[
                {
                    "role": "system",
                    "content": """You are a medical query validator.
                                Answer ONLY with 'YES' or 'NO'.
                                YES = input contains SPECIFIC medical information:
                                diagnosed condition, medications, lab values,
                                or detailed symptoms with context
                                NO  = vague symptoms alone (tired, headache),
                                gibberish, non-medical content,
                                or insufficient detail for trial matching"""
                },
                {
                    "role": "user",
                    "content": f"Is this a medical query? {query}"
                }
            ],
            max_tokens=5,
            # max_tokens=5 because we only need "YES" or "NO" — nothing more
            # this keeps this call extremely fast and cheap
            temperature=0.0
            # temperature=0 = deterministic
            # always same answer for same input
        )
        
        answer = response.choices[0].message.content.strip().upper()
        return "YES" in answer
    
    except Exception:
        # If validation call fails for any reason
        # default to true,don't block real users due to API errors
        return True

if search_clicked:
    
    # Validate input — don't search if empty
    if not patient_query.strip():
        st.warning("⚠️ Please describe your medical condition before searching.")
    
    elif len(patient_query.strip()) < 20:
        st.warning("⚠️ Please provide more detail about your condition for better results.")
    
    elif not is_valid_medical_query(patient_query):
        st.error(
            "⚠️ Your description doesn't appear to contain "
            "medical information. Please describe your medical "
            "condition, age, symptoms, and current medications."
        )
    else:
        # Step 1 -: Retrieve matching trials
        with st.spinner("🔍 Searching through clinical trials..."):
            try:
                trials = retrieve_trials(
                    patient_query,
                    n_results=n_results
                )
                if not trials:
                    st.warning(
                        "⚠️ No relevant clinical trials found for your description. "
                        "Please describe a specific medical condition, age, "
                        "current medications, and symptoms."
                            )
                    st.stop()
                    
            except Exception as e:
                st.error(f"❌ Retrieval error: {str(e)}")
                st.stop()
        

        st.success(f"✅ Found {len(trials)} matching trials. Generating analysis...")
        
        # Step 2 -: Show trial cards BEFORE LLM analysis
        st.subheader("📊 Matched Trials")
        
        for i, trial in enumerate(trials):
            with st.expander(
                f"Trial {i+1}: {trial['title']} — "
                f"Similarity: {trial['similarity_score']}%",
                expanded=True
            ):
                # three columns for trial metadata
                c1, c2, c3 = st.columns(3)
                
                with c1:
                    st.markdown(f"**🔬 NCT ID:** `{trial['nct_id']}`")
                    st.markdown(f"**📍 Location:** {trial['location']}")
                
                with c2:
                    st.markdown(f"**🏥 Condition:** {trial['condition']}")
                    st.markdown(f"**📅 Phase:** {trial['phase']}")
                
                with c3:
                    st.markdown(f"**✅ Status:** {trial['status']}")
                    st.markdown(f"**🔄 Updated:** {trial['last_updated']}")
                
                st.markdown("**📝 Summary:**")
                st.markdown(trial['summary'])
                
                # Collapsible eligibility section
                with st.expander("📋 View Full Eligibility Criteria"):
                    st.markdown(trial['eligibility'])
        
        # Step 3 -: Generate and show LLM analysis
        st.markdown("---")
        st.subheader("🤖 AI Match Analysis")
        st.markdown("*Powered by Llama 3.3 70B via Groq — "
                   "grounded in retrieved trial documents only*")
        
        with st.spinner("🧠 Llama 3 is analyzing your profile against each trial..."):
            try:
                explanation = generate_explanation(patient_query, trials)
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                st.stop()
        
        # Display the explanation in a container
        with st.container():
            st.markdown(explanation)
        
        # Step 4 -: ClinicalTrials.gov links
        st.markdown("---")
        st.subheader("🔗 View Trials on ClinicalTrials.gov")
        
        for trial in trials:
            trial_url = f"https://clinicaltrials.gov/study/{trial['nct_id']}"
            st.markdown(
                f"• [{trial['title']}]({trial_url}) — `{trial['nct_id']}`"
            )
        
        # Step 5 -: Reminder disclaimer at bottom
        st.markdown("---")
        st.markdown("""
        <div class="disclaimer">
        🏥 <strong>Next Steps:</strong> If you find a trial that interests you, 
        visit the ClinicalTrials.gov link above for complete information and 
        contact the research team directly. Always discuss participation 
        with your doctor first.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR — How it works
# ─────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ How It Works")
    
    st.markdown("""
    **1. You describe your condition**
    Type your symptoms, diagnosis, age, 
    medications and medical history.
    
    **2. Semantic Search**
    Your description is converted into 
    a medical vector using BiomedBERT — 
    trained on 21M medical papers.
    
    **3. ChromaDB Search**
    Your vector is compared against 
    500+ clinical trial embeddings 
    using cosine similarity.
    
    **4. AI Analysis**
    Llama 3.3 70B reads your profile 
    and the matched trials, then 
    explains why you match and what 
    to watch out for.
    
    **5. Grounded Results**
    Every statement is based only on 
    real trial documents — no 
    hallucinated information.
    """)
    
    st.markdown("---")
    st.header("📊 Database Info")
    st.metric("Total Trials", "10,000+")
    st.metric("Condition", "Multiple conditions")
    st.metric("Status", "Recruiting Only")
    st.metric("Source", "ClinicalTrials.gov")
    
    st.markdown("---")
    st.header("🛠️ Tech Stack")
    st.markdown("""
    - **Embeddings:** BiomedBERT
    - **Vector DB:** ChromaDB  
    - **LLM:** Llama 3.3 70B
    - **Inference:** Groq
    - **Framework:** LangChain
    - **UI:** Streamlit
    """)
    
    st.markdown("---")
    st.caption("Built with 🖤 using open source tools. 100% free.")

