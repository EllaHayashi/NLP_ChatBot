# scripts/04_finetune.py

#this code fine tunes the DistilBert model for QA using the cleaned squad format data
#we want to train a AQ model that takes a question and a context to learn to extract the answere
# we want the QA model to be trained on the Question and answeres that we made and use
#back propegation to adjust the weights
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""

# load distilbert model and tokenizer
model_name = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# use cleaned squad json file for dataset
dataset = load_dataset("json", data_files={"train": "data/squad/clean_train.json"}, field="data")

# flatten squad format into indaviduals
#we take nested squad format into rows
#the rows are context, question, answere and answere start
samples = []
for item in dataset["train"]:
    for paragraph in item["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            answer = qa["answers"][0]
            answer_text = answer["text"]
            answer_start = answer["answer_start"]
            if context[answer_start:answer_start + len(answer_text)] != answer_text:
                continue
            samples.append({
                "context": context,
                "question": question,
                "answer_text": answer_text,
                "answer_start": answer_start
            })

dataset = Dataset.from_list(samples)

# split the data into train and test
split_dataset = dataset.train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# we have to preprocess each one for training
# to do this we tokenize the question-context pair
# find which tokens align with start and end of answere span
# we need to re do the 'starts' number to match the new tokens so we remake this one
# add a start position and end position for where the toekn aligns
def preprocess(example):
    tokenized = tokenizer(
        example["question"],
        example["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True
    )

    start_char = example["answer_start"]
    end_char = start_char + len(example["answer_text"])
    offsets = tokenized["offset_mapping"]

    start_token = end_token = None

    for idx, (start, end) in enumerate(offsets):
        if start <= start_char < end:
            start_token = idx
        if start < end_char <= end:
            end_token = idx

    # if it doesn't find it just give it 0 and 0 
    if start_token is None or end_token is None:
        start_token = end_token = 0

    tokenized.update({
        "start_positions": start_token,
        "end_positions": end_token
    })
    tokenized.pop("offset_mapping")

    return tokenized

# apply the preprocessing to the train and test
train_tokenized = train_dataset.map(preprocess, remove_columns=train_dataset.column_names)
eval_tokenized = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names)

# set up the training config
training_args = TrainingArguments(
    output_dir="models/distilbert-qa",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="logs",
    logging_steps=50,
    num_train_epochs=4,
    per_device_train_batch_size=8,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none"
)

# set up trainer api
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    tokenizer=tokenizer
)

# train
print("fine-tune...")
trainer.train()
#save the fine tuned model
tokenizer.save_pretrained("models/distilbert-qa")
print("\nmodels/distilbert-qa")