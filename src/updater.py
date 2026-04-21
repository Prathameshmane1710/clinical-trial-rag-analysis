import json
import os
import requests
import time
# import schedule
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# SETUP — same model and DB as embedder.py

# ─────────────────────────────────────────

print("Initializing updater...")

# Path where we save the last run date
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAST_RUN_FILE = os.path.join(BASE_DIR, "data", "last_update.json")

model = SentenceTransformer(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
)

client = chromadb.PersistentClient(path="embeddings/chroma_db")
collection = client.get_collection("clinical_trials")

print("Updater initialized and ready!")


def get_last_run_date():
    if os.path.exists(LAST_RUN_FILE):
        with open(LAST_RUN_FILE, "r") as f:
            data = json.load(f)
            last_run = data.get("last_run_date")
            print(f"Last successful run: {last_run}")
            return last_run
    
    # First ever run -: use yesterday as default
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"No previous run found. Using yesterday: {yesterday}")
    return yesterday


def save_last_run_date():
    os.makedirs(os.path.dirname(LAST_RUN_FILE), exist_ok=True)
    
    today = datetime.today().strftime("%Y-%m-%d")
    
    with open(LAST_RUN_FILE, "w") as f:
        json.dump({
            "last_run_date": today,
            "last_run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"Saved last run date: {today}")

def fetch_changed_trials(since_date, max_records=50000):
    
    changed_trials = []
    next_page_token = None
    
    while True:

        # Safety cap -: stop if we hit the limit
        if len(changed_trials) >= max_records:
            print(f"Hit safety cap of {max_records} records")
            print("Run the script again to continue")
            break
        
        params = {
            # AREA[LastUpdatePostDate] tells the API
            # to filter by the last update date field
            # RANGE[date, MAX] means from date until now
            "filter.advanced": f"AREA[LastUpdatePostDate]RANGE[{since_date}, MAX]",
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
            
        changed_trials.extend(studies)
        print(f"Fetched {len(changed_trials)} changed trials so far...")
        
        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break
            
        time.sleep(0.5)
    
    print(f"Total changed trials found: {len(changed_trials)}")
    return changed_trials


def clean_eligibility_text(text):
    if not text:
        return ""
    
    import re
    text = text.replace("\\>", ">").replace("\\<", "<").replace("\\*", "")
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace("* ", "").strip()
    return text


def parse_trial(trial):
    """
    Same parsing logic as parser.py
    Extracts clean flat dict from nested JSON
    Written again to run independently in updater.py without importing parser.py
    """
    try:
        protocol = trial["protocolSection"]
        
        nct_id = protocol.get(
            "identificationModule", {}
        ).get("nctId", "")
        
        title = protocol.get(
            "identificationModule", {}
        ).get("briefTitle", "")
        
        status = protocol.get(
            "statusModule", {}
        ).get("overallStatus", "")
        
        last_updated = protocol.get(
            "statusModule", {}
        ).get("lastUpdatePostDateStruct", {}).get("date", "")
        
        summary = protocol.get(
            "descriptionModule", {}
        ).get("briefSummary", "")
        
        conditions = protocol.get(
            "conditionsModule", {}
        ).get("conditions", [])
        condition = conditions[0] if conditions else ""
        
        phases = protocol.get(
            "designModule", {}
        ).get("phases", [])
        phase = phases[0] if phases else "Not specified"
        
        eligibility_raw = protocol.get(
            "eligibilityModule", {}
        ).get("eligibilityCriteria", "")
        eligibility = clean_eligibility_text(eligibility_raw)
        
        # Extract location
        try:
            locations = protocol.get(
                "contactsLocationsModule", {}
            ).get("locations", [])
            if locations:
                city = locations[0].get("city", "")
                country = locations[0].get("country", "")
                location = f"{city}, {country}" if city else country
            else:
                location = "Location not specified"
        except:
            location = "Location not specified"
        
        if not nct_id or not eligibility:
            return None
        
        return {
            "nct_id": nct_id,
            "title": title,
            "status": status,
            "last_updated": last_updated,
            "summary": summary,
            "condition": condition,
            "phase": phase,
            "eligibility": eligibility,
            "location": location,
            "text_to_embed": f"Title: {title}. Condition: {condition}. Eligibility: {eligibility}"
        }
    
    except Exception as e:
        print(f"Parse error for trial: {e}")
        return None


def delete_trial_vectors(nct_id):
    """
    Deletes all chunk of vectors belonging to a single trial
    one trial can have multiple chunks:
    NCT04123456_chunk_0
    NCT04123456_chunk_1
    NCT04123456_chunk_2

    """
    try:
        # Find all chunks for this trial
        existing = collection.get(
            where={"nct_id": nct_id}
        )
        
        if existing["ids"]:
            # Delete all of them at once
            collection.delete(ids=existing["ids"])
            print(f"  Deleted {len(existing['ids'])} "
                  f"vectors for {nct_id}")
            return len(existing["ids"])
        return 0
        
    except Exception as e:
        print(f"  Delete error for {nct_id}: {e}")
        return 0


def add_trial_vector(trial):
    """
    Embeds a single trial and adds to ChromaDB
    Used for both NEW trials and UPDATED trials
    (after deleting old vectors for updated ones)
    """
    try:
        text = trial["text_to_embed"]
        
        # Embed using BiomedBERT
        embedding = model.encode(text).tolist()
        
        metadata = {
            "nct_id": trial["nct_id"],
            "title": trial["title"],
            "status": trial["status"],
            "condition": trial["condition"],
            "phase": trial["phase"],
            "location": trial["location"],
            "last_updated": trial["last_updated"],
            "summary": trial["summary"],
            "eligibility": trial["eligibility"]
        }
        
        collection.add(
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[f"{trial['nct_id']}_chunk_0"]
        )
        return True
        
    except Exception as e:
        print(f"  Add error for {trial['nct_id']}: {e}")
        return False


def run_daily_update():
    
    print("\n" + "="*50)
    print(f"DAILY UPDATE STARTED")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    
    # Step 1 -: Get yesterday's date
    since_date = get_last_run_date()

    print(f"Fetching all changes since: {since_date}")
    # Step 2 -: Fetch changed trials from API
    changed_trials_raw = fetch_changed_trials(since_date)
    
    if not changed_trials_raw:
        print("No changes found today. Database is up to date!")
        save_last_run_date()
        return
    
    # Step 3 -: Parse each changed trial
    # Track statistics for logging
    stats = {
        "new": 0,
        "updated": 0,
        "closed": 0,
        "skipped": 0,
        "errors": 0
    }
    
    for raw_trial in changed_trials_raw:
        
        parsed = parse_trial(raw_trial)
        
        if not parsed:
            stats["skipped"] += 1
            continue
        
        nct_id = parsed["nct_id"]
        status = parsed["status"]
        
        # Check if this trial already exists in ChromaDB
        # by looking for its vector using metadata filter
        existing = collection.get(
            where={"nct_id": nct_id}
        )
        already_exists = len(existing["ids"]) > 0
        
        # CASE 1: Trial is closed/completed
        # Remove it from ChromaDB entirely
        # We don't want to match patients to
        # trials they can't join anymore
        if status in ["COMPLETED", "TERMINATED", 
                      "WITHDRAWN", "SUSPENDED"]:
            if already_exists:
                delete_trial_vectors(nct_id)
                print(f"  ❌ CLOSED: {nct_id} — removed")
                stats["closed"] += 1
            else:
                stats["skipped"] += 1
        
        # CASE 2: Trial exists and was updated
        # Delete old vectors first, then re-embed
        # ChromaDB doesn't have an "update" operation
        # for vectors — you must delete then re-add
        elif already_exists:
            delete_trial_vectors(nct_id)
            success = add_trial_vector(parsed)
            if success:
                print(f"  🔄 UPDATED: {nct_id}")
                stats["updated"] += 1
            else:
                stats["errors"] += 1
        
        # CASE 3: Completely new trial
        # Just embed and add
        else:
            success = add_trial_vector(parsed)
            if success:
                print(f"  ✅ NEW: {nct_id}")
                stats["new"] += 1
            else:
                stats["errors"] += 1
    
    save_last_run_date()
    
    # Step 4 -: Print summary
    print("\n" + "="*50)
    print("DAILY UPDATE COMPLETE")
    print(f"✅ New trials added:    {stats['new']}")
    print(f"🔄 Trials updated:     {stats['updated']}")
    print(f"❌ Trials removed:     {stats['closed']}")
    print(f"⏭️  Skipped:            {stats['skipped']}")
    print(f"🔥 Errors:             {stats['errors']}")
    print(f"📊 Total in ChromaDB:  {collection.count()}")
    print(f"Next update in 24 hours")
    print("="*50)


# def start_scheduler():
#     """
#     Sets up the daily schedule using
#     the 'schedule' library
    
#     """
#     print("Scheduler started!")
#     print("Daily update will run at 02:00 AM every day")
#     print("Running initial update now...")
    
#     # Run once immediately on startup to ensure DB is up to date
#     run_daily_update()
    
#     # Then schedule for every day at 2 AM
#     schedule.every().day.at("02:00").do(run_daily_update)
    
#     # Keep the script running forever
#     # checking every 60 seconds if it's time to run
#     print("\nScheduler running. Press Ctrl+C to stop.")
#     while True:
#         schedule.run_pending()
#         time.sleep(60)
#         # check every 60 seconds if a job is due
#         # not every millisecond — saves CPU


if __name__ == "__main__":
    run_daily_update()