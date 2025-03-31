# scripts/03_clean_data.py

#the purpose of this is to do data cleaning on the squad formatted QA pairs
#this script uses the train.json file in squad directory and converts it
#into clean_train.json
import json
from transformers import AutoTokenizer

MAX_ANSWER_TOKENS = 30
# we are also using distilbert to tokenize the squad formatted qa file
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
#open the uncleaned squad file
with open("data/squad/train.json", "r", encoding="utf-8") as f:
    #load the data from it
    data = json.load(f)

#setup for cleaning
cleaned_data = {"data": []}
kept_count = 0
skipped_count = 0

#iterate threw uncleaned data
for item in data["data"]:
    #keep title
    title = item["title"]
    cleaned_paragraphs = []

    #loop threw each paragraph
    for para in item["paragraphs"]:
        #extracts the context and loops over pairs
        context = para["context"]
        qas = []
        #loop threw each pair and do cleaning
        #check for answere and context
        #check for length
        #check for context spread
        for qa in para["qas"]:
            answer_text = qa["answers"][0]["text"]
            answer_start = qa["answers"][0]["answer_start"]

            # the answeres must match the context
            if context[answer_start:answer_start + len(answer_text)] != answer_text:
                skipped_count += 1
                continue

            # shouldn't be larger then 30 tokens
            if len(tokenizer.tokenize(answer_text)) > MAX_ANSWER_TOKENS:
                skipped_count += 1
                continue

            # if the answere starts at the very end of the context then skip
            if answer_start >= len(context) - 15:
                skipped_count += 1
                continue

            # if it passes the cleaning requirments then append to cleaned
            qas.append(qa)
            kept_count += 1
        #append to cleaned paragraphs
        if qas:
            cleaned_paragraphs.append({
                "context": context,
                "qas": qas
            })

    if cleaned_paragraphs:
        cleaned_data["data"].append({
            "title": title,
            "paragraphs": cleaned_paragraphs
        })

# save it in the new cleaned json
with open("data/squad/clean_train.json", "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, indent=2)

print(f"Done! Kept {kept_count} QAs, skipped {skipped_count}. Cleaned file saved to data/squad/clean_train.json")
