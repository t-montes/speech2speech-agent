import google.generativeai as genai
from .llm import LLM
import pickle
import numpy
import time
import json
import os
import re

pickle_path = "knowledge_base/{}.pkl"
json_path = "knowledge_base/{}.json"
os.makedirs("knowledge_base", exist_ok=True)

get_qa_prompt = """\
Given the following document, provide a list of all the question and answer pairs present.
The output should have the following format:
[
    {"question": "What is the latest you can give me a call back?", "answer": "We are open until 5 pm. Would you like a call around 4:30 pm?"},
    ...
]

State the question and answers exactly as they appear in the document.
Don't include anything that is not in the document, and don't make any assumptions.\
"""

def is_cached(path, cache_time):
    current_time = time.time()
    if not os.path.exists(path):
        return False
    
    file_time = os.stat(path).st_ctime
    return (current_time - file_time) < cache_time

def parse_json(r: str) -> dict:
    match = re.search(r'```json(.*?)```', r, re.DOTALL)
    if not match:
        match = re.search(r'```(.*?)```', r, re.DOTALL)
    if not match:
        json_str = r.strip()
    else:
        json_str = match.group(1).strip()

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON found in the input string: {e}")

class KnowledgeBase():
    def __init__(self, api_key, faq_document_path, embedding_model="models/text-embedding-004", cache_time=3600, force_update=False):
        self.llm = LLM(api_key, temperature=0)
        self.embedding_model = embedding_model

        file_name = os.path.splitext(os.path.basename(faq_document_path))[0]
        self.pickle_path = pickle_path.format(file_name)
        self.json_path = json_path.format(file_name)

        if force_update or not is_cached(self.pickle_path, cache_time):
            self.create_knowledge_base(faq_document_path)
        else: print("Cached!")
        
        with open(self.pickle_path, "rb") as f:
            self.embeddings = pickle.load(f)
        
        with open(self.json_path, "r") as f:
            self.qa_list = json.load(f)
    
    def embed(self, *texts):
        embeddings = genai.embed_content(
            self.embedding_model, content=list(texts)
        )['embedding']
        return numpy.array(embeddings)
    
    def create_knowledge_base(self, faq_document_path):
        qa_text = self.llm(get_qa_prompt, faq_document_path)
        qa_list = parse_json(qa_text)

        with open(self.json_path, "w") as f:
            json.dump(qa_list, f)
        
        questions = [qa["question"] for qa in qa_list]
        embeddings = self.embed(*questions)

        with open(self.pickle_path, "wb") as f:
            pickle.dump(embeddings, f)
    
    def __call__(self, query):
        query_embedding = self.embed(query)
        similarities = (self.embeddings @ query_embedding.T).flatten()
        max_index = similarities.argmax()
        return self.qa_list[max_index]["answer"]
