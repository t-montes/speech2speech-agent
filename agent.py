from components import STT, LLM, TTS, KnowledgeBase, replace_placeholders
from dotenv import load_dotenv
import os
load_dotenv()

# Variables and Hyperparameters
google_key = os.getenv("GOOGLE_API_KEY")
eleven_key = os.getenv("ELEVENLABS_API_KEY")
faq_file = os.getenv("FAQ_FILE")

min_similarity = 0.75

placeholders = {
    "Revenue Partner": "Lifemart",
    "User Name": "Camila",
    "User Email": "camila@gmail.com",
    "User Phone": "1 914 365 138"
}

agent_system_instruction = """\
You are an assistant for Domu, the insurance partner of [Revenue Partner]. You are on a call with a user named [User Name]. 
Your task is to answer the user's questions based on the provided history, the context, and your specific capabilities.

Context:
- The purpose of the call is to assist the user in finding the right insurance and confirming their email and phone number.
- After confirming these details, the user will be transferred to a licensed agent.
- Always base your responses on the history and context. Do not provide information you cannot verify.
- Since this is a call, end each response with a polite yes/no question, such as: "Would you like to continue?" or "Did I answer your question?"
"""

agent_system_instruction = replace_placeholders(placeholders, agent_system_instruction)

classify_answer_instruction = """\
Given the human answer, please return ONLY one of the following possible types:
- 'yes': Overall, the answer is ONLY answering positively.
- 'no': Overall, the answer is ONLY answering negatively.
- 'other': Neither 'yes' nor 'no'.

Don't give explanation, only output the type.\
"""

state_machine = {
    "step_1": {
        "question": "Hi [User Name], this is Bella from Domu. We are the insurance partner of [Revenue Partner] and we noticed that you quoted on our website and it seems you are seeking help in finding the right insurance for your car. Is that correct?",
        "type": "question",
        "yes": "step_2",
        "no": "not_interested",
        "other": "step_2"
    },
    "step_2": {
        "question": "I just need to confirm that your email is [User Email] and your phone number is [User Phone]. Are those correct?",
        "type": "question",
        "yes": "step_3",
        "no": "update_information",
        "other": "step_2" # Stay in the same step for non-scripted questions
    },
    "step_3": {
        "question": "Perfect, I'll be transferring you to one of our licensed agents to help you find the right insurance for you.",
        "type": "finish",
        "yes": None,
        "no": None,
        "other": None
    },
    "not_interested": {
        "question": "I see. Can you tell me why you are not looking for insurance right now?",
        "type": "listen_and_finish",
        "yes": None,
        "no": None,
        "other": None
    },
    "update_information": {
        "question": "Understood. I'll transfer you to one of our licensed agents, and they will help you update your information.",
        "type": "finish",
        "yes": None,
        "no": None,
        "other": None
    }
}

for k in state_machine:
    state_machine[k]["question"] = replace_placeholders(placeholders, state_machine[k]["question"])

intermediate_question_no = "Then, what else can I help you with?"
end_states = ["finish", "transfer"]
state = "step_1"

history = []
def add_to_history(role, content):
    global state
    print(f"({state}) {role}: {content}")
    history.append({"role": role, "state": state, "content": content})

# Initialize components
stt = STT(do_print=True)
tts = TTS(eleven_key)
answer_llm = LLM(google_key, system_instruction=agent_system_instruction)
classify_answer_llm = LLM(google_key, system_instruction=classify_answer_instruction)
kbase = KnowledgeBase(google_key, faq_file, placeholders=placeholders)

os.system('cls' if os.name == 'nt' else 'clear')

# Main loop
while state not in end_states:
    question = state_machine[state]["question"]
    q_type = state_machine[state]["type"]
    add_to_history("agent", question)
    tts(question)

    intermediate_question = ""
    match (q_type):
        case "question":
            answer = stt()
            add_to_history("user", answer)
            answer_type = classify_answer_llm(answer).strip()

            state = state_machine[state][answer_type]

            if answer_type == "other":
                while answer_type != "yes":
                    if answer_type == "other":
                        intermediate_question = answer

                        score, answer = kbase(intermediate_question)
                        if score > min_similarity:
                            add_to_history("agent", answer)
                            tts(answer)
                        else:
                            answer = answer_llm(intermediate_question)
                            add_to_history("agent", answer)
                            tts(answer)

                    elif answer_type == "no":
                        intermediate_question = intermediate_question_no
                        add_to_history("agent", intermediate_question)
                        tts(intermediate_question)
                    
                    else:
                        print(f"Invalid answer type!! ({answer_type})")

                    answer = stt()
                    add_to_history("user", answer)
                    answer_type = classify_answer_llm(answer).strip()

        case "finish":
            state = "finish"
        case "listen_and_finish":
            answer = stt()
            add_to_history("user", answer)
            state = "finish"
        case _:
            pass

# End of the conversation
import datetime
import json
import os

os.makedirs("history", exist_ok=True)

history_file = f"history/chat_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
with open(history_file, "w") as f:
    json.dump(history, f, indent=2)
