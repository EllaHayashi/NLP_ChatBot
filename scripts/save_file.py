from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# 路径要和训练时一致
model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")

model.save_pretrained("models/distilbert-qa")
tokenizer.save_pretrained("models/distilbert-qa")

print("模型和 tokenizer 已保存到 models/distilbert-qa")
