from fastapi import FastAPI
from pydantic import BaseModel
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from fastapi.middleware.cors import CORSMiddleware

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

app = FastAPI()

# Load trained model and necessary files
model = load_model('chatbot_model.keras')
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

class Message(BaseModel):
    message: str

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Cấu hình CORSMiddleware để cho phép CORS từ mọi nguồn (*)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

@app.post("/chatbot")
async def chatbot(msg: Message):
    ints = predict_class(msg.message, model)
    res = get_response(ints, intents)
    return {"response": res}
