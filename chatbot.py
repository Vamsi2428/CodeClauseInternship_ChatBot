print("1.normal chatting")
print("2.Output text to speech")
print("3.text to speech and speech to text conversation")
ch=int(input("enter value 1,2 or 3:"))
#print("if you want to change the mode of conversation ")
import json
import string
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout

nltk.download("punkt")
nltk.download("wordnet")
#open and load json file
f=open("data.json")
data=json.load(f)
lemmatizer=WordNetLemmatizer()

words=[]
classes=[]
doc_x=[]
doc_y=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens=nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    if intent["tag"] not in classes:
        classes.append(intent["tag"])
words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]
words[:10]
words=sorted(set(words))
classes=sorted(set(classes))


training=[]
out_empty=[0]*len(classes)

# creating a bag of words model

for idx, doc in enumerate(doc_x):
    bow=[]
    text=lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)
    output_row=list(out_empty)
    output_row[classes.index(doc_y[idx])]=1

    training.append([bow, output_row])

random.shuffle(training)

training=np.array(training,dtype=object)

train_X=np.array(list(training[:,0]))
train_y=np.array(list(training[:,1]))

input_shape=(len(train_X[0]),)
output_shape=len(train_y[0])

epochs=500

model=Sequential()
model.add(Dense(128, input_shape=input_shape,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(output_shape,activation='softmax'))
adam=tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
# legacy -->tf.keras.optimizers.legacy.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
             optimizer="adam",
             metrics="accuracy")

print(model.summary())

model.fit(x=train_X, y=train_y, epochs=500, verbose=1)



def clean_text(text):
    tokens=nltk.word_tokenize(text)
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def bag_of_words(text,vocab):
    tokens=clean_text(text)
    bow=[0]*len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word==w:
                bow[idx]=1
    return np.array(bow)


def pred_class(text, vocab,labels):
    bow=bag_of_words(text, vocab)
    result=model.predict(np.array([bow]))[0]
    thresh=0.2
    y_pred=[[idx,res] for idx, res in enumerate(result) if res>thresh]

    y_pred.sort(key=lambda x:x[1], reverse=True)
    return_list=[]
    for r in y_pred:
        return_list.append(labels[r[0]])
    return return_list

def get_response(intents_list, intents_json):
    tag=intents_list[0]
    list_of_intents=intents_json["intents"]
    for i in list_of_intents:
        if i["tag"]==tag:
            result=random.choice(i["responses"])
            break
    return result

while True:
    if ch==1:
        message=input("")
        intents=pred_class(message, words, classes)
        result=get_response(intents,data)
        print(result)
    elif ch==2:
        message=input("")
        intents=pred_class(message, words, classes)
        result=get_response(intents,data)
        print(result)
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(result)
        engine.runAndWait()
    elif ch==3:
        import speech_recognition as sr
        import pyttsx3 
        r = sr.Recognizer() 
        def SpeakText(command):
                
                engine = pyttsx3.init()
                engine.say(command) 
                engine.runAndWait()
                

        while(1): 
                
                try:
                    with sr.Microphone() as source2:
                        SpeakText("please speak")
                        r.adjust_for_ambient_noise(source2, duration=0.2)
                        audio2 = r.listen(source2)
                        MyText = r.recognize_google(audio2)
                        MyText = MyText.lower()
                        message=MyText
                        intents=pred_class(message, words, classes)
                        result=get_response(intents,data)
                        print(result)
                        SpeakText(result)
                                                
                                                    
                except sr.RequestError as e:
                        print("Could not request results; {0}".format(e))
                        
                except sr.UnknownValueError:
                        print("Could not recognize please speak clearly.")
                        
    else:
        ch=1
    if message=="exit" or message=="bye":
        print("Bye,see you sooon")
        break
        














		

