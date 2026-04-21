import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

PROMPT_TEMPLATE = """
You are a clinical trial matching assistant helping patients 
find relevant medical trials.

PATIENT PROFILE:
{patient_query}

MATCHED CLINICAL TRIALS:
{trials_text}

For each trial above, provide a structured analysis:

1. TRIAL NAME and ID
2. MATCH REASON: Why does this patient likely qualify? 
   Be specific — reference exact eligibility criteria.
3. WATCH OUT: What criteria might disqualify this patient?
   Only mention if there is a genuine concern.
4. LOCATION: Where is this trial being conducted?
5. TREATMENT: What intervention would the patient receive?

IMPORTANT RULES:
- Only use information present in the trial details above
- Do not make up or assume any medical information
- Do not provide medical advice or diagnosis
- If eligibility is unclear, say so honestly
- Keep each trial analysis concise and clear

Begin your analysis:
"""


def format_trials_for_prompt(trials):
    """
    fun converts list of trial dicts into
    clean readable text for the prompt
    """
    
    trials_text = ""
    
    for i, trial in enumerate(trials):
        trials_text += f"""
TRIAL {i+1}:
- ID: {trial['nct_id']}
- Title: {trial['title']}
- Condition: {trial['condition']}
- Phase: {trial['phase']}
- Location: {trial['location']}
- Status: {trial['status']}
- Similarity Score: {trial['similarity_score']}%
- Summary: {trial['summary']}
- Eligibility Criteria: {trial['eligibility']}
---"""
    
    return trials_text


def generate_explanation(patient_query, trials):
    """
    takes patient query + retrieved trials
    sends them to Groq Llama 3
    returns human readable explanation
    """
    
    # Step 1 -: Format trials into readable text
    trials_text = format_trials_for_prompt(trials)
    
    # Step 2 -: Fill in the prompt template
    filled_prompt = PROMPT_TEMPLATE.format(
        patient_query=patient_query,
        trials_text=trials_text
    )
    
    print("Sending to Groq Llama 3 for reasoning...")
    
    # Step 3 -: Send to Groq
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[
            {
                "role": "system",
                "content": """You are a helpful clinical trial matching assistant. 
                CRITICAL RULE: If the patient profile does not contain 
                clear medical information (diagnosis, symptoms, age, 
                medications), respond with exactly:
                "I cannot find relevant trials without a proper medical 
                profile. Please provide your diagnosis, age, symptoms, 
                and current medications."

                Do not attempt to match vague or non-medical descriptions
                to clinical trials under any circumstances."""
            },
            {
                "role": "user",
                "content": filled_prompt
            }
        ],
        temperature=0.1,
        # temperature controls randomness:
        # 0.0 = completely deterministic, same answer every time
        # 1.0 = very creative/random
        # 0.1 = mostly consistent with slight variation
        # For medical matching low temperature
        # because it provides consistent, factual answers not creative ones
        max_tokens=2000
        # maximum length of response
        # 2000 tokens ≈ ~1500 words
        # enough for detailed analysis of 3-5 trials
    )
    
    # Step 4 -: Extract text from response
    explanation = response.choices[0].message.content
    
    return explanation


if __name__ == "__main__":
    
    from retriever import retrieve_trials
    
    patient_query = "45 year old male with type 2 diabetes, HbA1c 8.2, metformin stopped working, no insulin use"
    
    print("="*50)
    print("CLINICAL TRIAL MATCHER — FULL PIPELINE TEST")
    print("="*50)
    
    print("\nStep 1: Retrieving matching trials...")
    trials = retrieve_trials(patient_query, n_results=3)
    print(f"Found {len(trials)} matching trials:")
    for t in trials:
        print(f"  - {t['title']} ({t['similarity_score']}%)")
    
    print("\nStep 2: Generating explanation with Groq Llama 3...")
    explanation = generate_explanation(patient_query, trials)
    
    print("\n" + "="*50)
    print("CLINICAL TRIAL MATCH ANALYSIS")
    print("="*50)
    print(explanation)
    print("\nPhase 4 complete!")