import os
import requests
import time
import json
def download_trials(condition, max_trials=1500, 
                    save_path="data/trials_raw.json"):
    
    print(f"Downloading: {condition}")
    
    # Load existing trials if file already exists
    # add to existing data
    # instead of overwriting it
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            all_trials = json.load(f)
        print(f"Loaded {len(all_trials)} existing trials")
    else:
        all_trials = []
    
    # Track existing NCT IDs to avoid duplicates
    existing_ids = {
        t["protocolSection"]["identificationModule"]["nctId"]
        for t in all_trials
        if "protocolSection" in t
    }
    
    new_trials = []
    next_page_token = None

    while True:
        params = {
            "query.cond": condition,
            "filter.overallStatus": "RECRUITING",
            "pageSize": 100,
            "format": "json",
            "fields": "NCTId,BriefTitle,BriefSummary,EligibilityCriteria,OverallStatus,Condition,LocationCity,LocationCountry,Phase,LastUpdatePostDate"
        }

        if next_page_token:
            params["pageToken"] = next_page_token

        try:
            response = requests.get(
                "https://clinicaltrials.gov/api/v2/studies",
                params=params,
                timeout=30
            )
            response.raise_for_status()
            data = response.json()

        except requests.exceptions.RequestException as e:
            print(f"API error: {e}")
            break

        studies = data.get("studies", [])
        if not studies:
            break

        # Only add trials we don't already have
        for trial in studies:
            try:
                nct_id = trial["protocolSection"]["identificationModule"]["nctId"]
                if nct_id not in existing_ids:
                    new_trials.append(trial)
                    existing_ids.add(nct_id)
            except KeyError:
                continue

        print(f"Condition '{condition}': {len(new_trials)} new trials found")

        if len(new_trials) >= max_trials:
            break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

        time.sleep(0.5)

    # Combine existing + new trials
    all_trials.extend(new_trials)

    # Save combined result
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_trials, f, indent=2, ensure_ascii=False)

    print(f"Condition '{condition}' done. Total trials in file: {len(all_trials)}")
    return all_trials


if __name__ == "__main__":
    conditions = [
        "diabetes",
        "cancer",
        "hypertension",
        "alzheimer",
        "asthma",
        "heart disease",
        "obesity",
        "depression"
    ]
    
    for condition in conditions:
        download_trials(
            condition=condition,
            max_trials=1500
        )
    
    # final count
    with open("data/trials_raw.json", "r") as f:
        final = json.load(f)
    print(f"\nFinal total unique trials: {len(final)}")