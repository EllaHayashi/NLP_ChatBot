# scripts/02_generate_qa.py
import os
import json
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering

# variables
INPUT_FOLDER = "data/raw"
OUTPUT_FILE = "data/squad/train.json"
QUESTION_GEN_MODEL = "google/flan-t5-small"
ANSWER_MODEL = "distilbert-base-cased-distilled-squad"

# load spacy model for sentence splitting
nlp = spacy.load("en_core_web_sm")
nlp.max_length = 10_000_000

# load DistilBERT model to extract answers from the passage based on generated question
print("Loading QA model...")
qa_pipeline = pipeline("question-answering", model=ANSWER_MODEL, tokenizer=ANSWER_MODEL, device=-1)

# load qa T5 model (google/flan) for generating questions from a passage
print("Loading question generation model...")
qgen_pipeline = pipeline("text2text-generation", model=QUESTION_GEN_MODEL, tokenizer=QUESTION_GEN_MODEL, device=-1)

# initialize a python dictionary that will save SQuAD-style JSON file
# we need to do this to train BERT
data = {"data": []}

# function using t5 model to generate teh question
# paramaters
#   text_chunk (from raw)
def generate_question_t5(text_chunk):
    prompt = f"Generate a question from the following passage:\n\n{text_chunk.strip()}"
    try:
        #use the t5 model to generate question given a prompt
        response = qgen_pipeline(prompt, max_new_tokens=64, do_sample=False)[0]["generated_text"]
        return response.strip()
    except Exception as e:
        print("Failed to generate question:", e)
        return None

# we store our QA in the squad format for future use
def build_squad_record(context, question, answer_dict, idx):
    answer_text = answer_dict['answer']
    start_idx = answer_dict['start']
    if not answer_text or start_idx == -1:
        print("Answer not found in context.")
        return None

    return {
        "context": context,
        "qas": [{
            "id": f"q{idx}",
            "question": question,
            "answers": [{"text": answer_text, "answer_start": start_idx}],
            "is_impossible": False
        }]
    }

# process each text file at the begining of execution
# read the text and split it into sentences using spacy
# chunk 100-500
idx = 0
for fname in os.listdir(INPUT_FOLDER):
    if fname.endswith(".txt"):
        # for every .txt file
        with open(os.path.join(INPUT_FOLDER, fname), "r", encoding="utf-8") as f:
            text = f.read()
            doc = nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]
            chunks, current = [], ""
            #chunking 100-500
            for sent in sentences:
                if len(current) + len(sent) < 500:
                    current += sent + " "
                else:
                    if len(current.strip()) >= 100:
                        chunks.append(current.strip())
                    current = sent + " "
            if len(current.strip()) >= 100:
                chunks.append(current.strip())

            # for each text chunk generate a question using t5 and Distilbert to extract answers
            # construct a SQUAD-format for records
            for i, chunk in enumerate(chunks):
                # we only have 500 chunks per file due to memory and not generate too many QA pairs per file
                if i >= 500:
                    break
                #display the current chunk
                print("Generating question for:", chunk[:80].replace("\n", " ") + "...")
                #uses the t5 function to generate the question
                question = generate_question_t5(chunk)
                if not question:
                    continue
                print("Question:", question)
                try:
                    # uses the DistilBERT to extract most relevant answers
                    # takes a question and the the original text to generate that question as context
                    result = qa_pipeline(question=question, context=chunk)
                    #output result found by DistilBERT
                    print("Answer:", result["answer"])
                    # change it into squad format and then save it in data
                    record = build_squad_record(chunk, question, result, idx)
                    if record:
                        data["data"].append({"title": fname, "paragraphs": [record]})
                        idx += 1
                    else:
                        print("Could not build a valid QA record")
                except Exception as e:
                    print(f"Failed to extract answer: {e}")

# writes the output data file 
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print(f"\nDone! Generated {idx} QA pairs. Output saved to {OUTPUT_FILE}")
