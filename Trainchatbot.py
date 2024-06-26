import nltk
nltk.download('punkt')  #phân đoạn từ
nltk.download('wordnet')  #chuẩn hóa từ

# import thư viện
from nltk.stem import WordNetLemmatizer
import json
import pickle

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

# Khởi tạo WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Đọc nội dung của tệp intents.json với mã hóa utf-8
with open('./intents.json', 'r', encoding='utf-8') as file:
    data_file = file.read()
intents = json.loads(data_file)

# Tiền xử lý dữ liệu
words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize mỗi từ
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # thêm mỗi cặp pattern và tag vào documents
        documents.append((w, intent['tag']))
        # thêm tag vào classes nếu chưa tồn tại
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize và chuyển đổi thành chữ thường, loại bỏ các từ không quan trọng
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Lưu words và classes vào các tệp pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Tạo dữ liệu huấn luyện
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([x[0] for x in training])
train_y = np.array([x[1] for x in training])

print("Training data created")

# Tạo model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện model và lưu
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('./chatbot_model.keras')

print("Model created and saved")
