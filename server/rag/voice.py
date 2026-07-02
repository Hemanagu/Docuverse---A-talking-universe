import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

def listen_to_user():

    with sr.Microphone() as source:

        print("Listening...")

        # adjust for background noise
        recognizer.adjust_for_ambient_noise(source, duration=1)

        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)

        print("You said:", text)

        return text

    except sr.UnknownValueError:
        print("Could not understand audio")
        return None

    except sr.RequestError:
        print("Speech service error")
        return None


def speak_answer(text):

    if text:   # speak only if text exists
        engine.say(text)
        engine.runAndWait()