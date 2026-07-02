from rag.voice import listen_to_user, speak_answer
from rag.rag_pipeline import run_rag_pipeline

COLLECTION_ID = "docuverse_kb"

def start_voice_chat():

    print("Voice assistant started. Say 'exit' to stop.")

    history = []

    while True:

        question = listen_to_user()
        if not question:
            continue

        if question.lower() == "exit":
            speak_answer("Goodbye")
            break

        print("User:", question)

        result = run_rag_pipeline(question, COLLECTION_ID, history)

        answer = result["answer"]

        print("AI:", answer)

        speak_answer(answer)

        history.append({"user": question})
        history.append({"assistant": answer})