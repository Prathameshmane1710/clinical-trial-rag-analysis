import json
import os
import re

def clean_eligibility_text(text):
    
    if not text:
        return ""
    
    # Removing markdown bullet points
    text = text.replace("\\>", ">")
    text = text.replace("\\<", "<")
    text = text.replace("\\*", "")
    
    # Removing excessive newlines and whitespace
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Cleaning up markdown artifacts
    text = text.replace("* ", "")
    text = text.strip()
    
    return text


def extract_location(trial):
    
    try:
        locations = trial["protocolSection"]["contactsLocationsModule"]["locations"]
        if locations:
            city = locations[0].get("city", "")
            country = locations[0].get("country", "")
            if city and country:
                return f"{city}, {country}"
            elif country:
                return country
            elif city:
                return city
    except (KeyError, IndexError):
        pass
    
    return "Location not specified"


def parse_single_trial(trial):
    
    try:
        protocol = trial["protocolSection"]
          
        nct_id = protocol.get("identificationModule", {}).get("nctId", "")
        
        title = protocol.get("identificationModule", {}).get("briefTitle", "")
        
        status = protocol.get("statusModule", {}).get("overallStatus", "")
        
        last_updated = protocol.get("statusModule", {}).get(
            "lastUpdatePostDateStruct", {}
        ).get("date", "")
        
        summary = protocol.get("descriptionModule", {}).get("briefSummary", "")
        
        conditions = protocol.get("conditionsModule", {}).get("conditions", [])
        condition = conditions[0] if conditions else ""
        
        phases = protocol.get("designModule", {}).get("phases", [])
        phase = phases[0] if phases else "Not specified"
        
        eligibility_raw = protocol.get("eligibilityModule", {}).get(
            "eligibilityCriteria", ""
        )
        eligibility_clean = clean_eligibility_text(eligibility_raw)
        
        location = extract_location(trial)
        
        # Skipping trials with no eligibility criteria
        # These are useless for our matching system
        if not eligibility_clean:
            return None
        
        # Skipping trials with no NCT ID
        if not nct_id:
            return None
        
        return {
            "nct_id": nct_id,
            "title": title,
            "status": status,
            "last_updated": last_updated,
            "summary": summary,
            "condition": condition,
            "phase": phase,
            "eligibility": eligibility_clean,
            "location": location,
            # This is the TEXT to be embedded into a vector
            # combine title + condition + eligibility
            # because all three together give the richest
            # semantic meaning for matching
            "text_to_embed": f"Title: {title}. Condition: {condition}. Eligibility: {eligibility_clean}"
        }
    
    except Exception as e:
        print(f"Error parsing trial: {e}")
        return None


def parse_all_trials(input_path="data/trials_raw.json", 
                     output_path="data/trials_parsed.json"):
    
    print(f"Loading raw trials from {input_path}...")
    
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_trials = json.load(f)
    
    print(f"Parsing {len(raw_trials)} trials...")
    
    parsed_trials = []
    skipped = 0
    
    for i, trial in enumerate(raw_trials):
        
        parsed = parse_single_trial(trial)
        
        if parsed:
            parsed_trials.append(parsed)
        else:
            skipped += 1
        
        # Progress updating every 100 trials
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(raw_trials)} trials...")
    
    # Saving parsed trials to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_trials, f, indent=2, ensure_ascii=False)
    
    print(f"Parsing complete.")
    print(f"Successfully parsed: {len(parsed_trials)} trials")
    print(f"Skipped (missing data): {skipped} trials")
    print(f"Saved to {output_path}")
    
    return parsed_trials


if __name__ == "__main__":
    parse_all_trials()