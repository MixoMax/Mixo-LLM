from transformers import AutoTokenizer
from transformers import GPT2LMHeadModel
from transformers import pipeline

model = GPT2LMHeadModel.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("model")

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)


input_text = input("Enter text: ")
while input_text.lower() != "q":
    output = generator(input_text)
    print(output[0]["generated_text"])
    print()
    input_text = input("Enter text: ")

