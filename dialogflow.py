import os
import dialogflow_v2
from google.api_core.exceptions import InvalidArgument
import pyttsx3
import speech_recognition as sr


DIALOGFLOW_PROJECT_ID = 'dopebot-kfnh'
DIALOGFLOW_LANGUAGE_CODE = 'en'
SESSION_ID = 'me'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'private_key.json'

session_client = dialogflow_v2.SessionsClient()
session = session_client.session_path(DIALOGFLOW_PROJECT_ID, SESSION_ID)
engine = pyttsx3.init()
engine.setProperty('rate', 150)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

r = sr.Recognizer()
while True:
    with sr.Microphone() as source:
        print("I am listening...")
        audio_text = r.listen(source)
        # print("Time over, thanks")
        # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
        try:
            # using google speech recognition
            input = str(r.recognize_google(audio_text))
            print(input)
            text_input = dialogflow_v2.types.TextInput(text=input, language_code=DIALOGFLOW_LANGUAGE_CODE)
            query_input = dialogflow_v2.types.QueryInput(text=text_input)
            try:
                response = session_client.detect_intent(session=session, query_input=query_input)
            except InvalidArgument:
                raise
            engine.say(response.query_result.fulfillment_text)
            engine.runAndWait()
        except:
            engine.say("Sorry, I did not hear you")

# print("Query text:", response.query_result.query_text)
# print("Detected intent:", response.query_result.intent.display_name)
# print("Detected intent confidence:", response.query_result.intent_detection_confidence)
# print("Fulfillment text:", response.query_result.fulfillment_text)

