import requests
import json
import os
import time

def download_trials(condition, max_trials=1000, save_path='data/trials_raw.json'):
    
    print(f"Starting download of clinical trials for condition: {condition}")

    # if os.path.exists(save_path):
    #     with open(save_path, "r", encoding="utf-8") as f:
    #         all_trials = json.load(f)
    #     print(f"Loaded {len(all_trials)} existing trials")
    # else:
    #     all_trials = []

    # # Track existing NCT IDs to avoid duplicates
    # existing_ids = {
    #     t["protocolSection"]["identificationModule"]["nctId"]
    #     for t in all_trials
    #     if "protocolSection" in t
    # }
    # new_trials = []

    all_trials = []
    next_page_token = None
    page_count = 0

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
            print("No more studies found.")
            break
        
        all_trials.extend(studies)
        page_count += 1
         # Only add trials we don't already have
        # for trial in studies:
        #     try:
        #         nct_id = trial["protocolSection"]["identificationModule"]["nctId"]
        #         if nct_id not in existing_ids:
        #             new_trials.append(trial)
        #             existing_ids.add(nct_id)
        #     except KeyError:
        #         continue
        print(f"Page {page_count} downloaded — {len(all_trials)} trials so far")

        if len(all_trials) >= max_trials:
            print(f"Reached max limit of {max_trials} trials")
            break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            print("No more pages available.")
            break
        
        time.sleep(0.5)

    # # Combine existing + new trials
    # all_trials.extend(new_trials)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_trials, f, indent=2, ensure_ascii=False)

    print(f"Download complete. {len(all_trials)} trials saved to {save_path}")
    return all_trials


if __name__ == "__main__":
    # conditions = [
    #     "diabetes",
    #     "cancer",
    #     "hypertension",
    #     "alzheimer",
    #     "asthma",
    #     "heart disease",
    #     "obesity",
    #     "depression"
    # ]
    download_trials(condition="diabetes", max_trials=500)

    # for condition in conditions:
    #     download_trials(
    #         condition=condition,
    #         max_trials=1500
    #     )
    
    # # Final count
    # with open("data/trials_raw.json", "r") as f:
    #     final = json.load(f)
    # print(f"\nFinal total unique trials: {len(final)}")















    



