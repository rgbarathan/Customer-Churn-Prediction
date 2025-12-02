
from transformers import pipeline

qa_pipeline = pipeline('question-answering', model='distilbert-base-uncased-distilled-squad')

context = "Comcast offers multiple plans including premium and basic packages."
question = "What plans does Comcast offer?"

answer = qa_pipeline(question=question, context=context)
print(f"Question: {question}")
print(f"Answer: {answer['answer']}")
