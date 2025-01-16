import google.generativeai as genai
import requests
import base64

def get_document(document_path):
    if document_path.startswith("http"):
        response = requests.get(document_path)
        document = response.content
    else:
        with open(document_path, "rb") as file:
            document = file.read()
    
    return {
        'mime_type': 'application/pdf',
        'data': base64.b64encode(document).decode("utf-8")
    }

class LLM():
    def __init__(self, api_key, model='gemini-1.5-flash', temperature=0.5):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model,
            generation_config={
                "temperature": temperature
            }
        )
    
    def __call__(self, prompt, document_path=None):
        ctx = []
        if document_path:
            document = get_document(document_path)
            ctx.append(document)
        ctx.append(prompt)
        return self.model.generate_content(ctx).text

if __name__ == '__main__':
    llm = LLM("...")
    response = llm("What's this document about?", "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf")
    print(response)

