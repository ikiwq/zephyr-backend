import datetime
import random

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import json
import pickle
import requests
from translate import Translator

class Zephyr():
    def __init__(self):
        with open("intents.json") as file:
            self.data = json.load(file)

        nltk.download('punkt')

        self.translator = Translator(from_lang="english", to_lang="italian")

        try:
            with open("data.pickle", "rb") as f:
                self.parole, self.labels, self.training, self.output = pickle.load(f)
        except:
            self.parole = []
            self.labels = []
            self.docs_x = []
            self.docs_y = []

            for intent in self.data["intents"]:
                for pattern in intent["patterns"]:
                    wrds = nltk.word_tokenize(pattern)
                    self.parole.extend(wrds)
                    self.docs_x.append(wrds)
                    self.docs_y.append(intent["tag"])

                if intent["tag"] not in self.labels:
                    self.labels.append(intent["tag"])

            self.parole = [stemmer.stem(w.lower()) for w in self.parole if w != "?"]
            self.parole = sorted(list(set(self.parole)))

            self.labels = sorted(self.labels)

            self.training = []
            self.output = []

            out_empty = [0 for _ in range(len(self.labels))]

            for x, doc in enumerate(self.docs_x):
                bag = []

                wrds = [stemmer.stem(w.lower()) for w in doc]

                for w in self.parole:
                    if w in wrds:
                        bag.append(1)
                    else:
                        bag.append(0)

                output_row = out_empty[:]
                output_row[self.labels.index(self.docs_y[x])] = 1

                self.training.append(bag)
                self.output.append(output_row)

            self.training = numpy.array(self.training)
            self.output = numpy.array(self.output)

            with open("data.pickle", "wb") as f:
                pickle.dump((self.parole, self.labels, self.training, self.output), f)

        net = tflearn.input_data(shape=[None, len(self.training[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(self.output[0]), activation="softmax")
        net = tflearn.regression(net)

        self.model = tflearn.DNN(net)

        self.model.fit(self.training, self.output, n_epoch=1000, batch_size=8)
        self.model.save("model.tflearn")

    def bag_of_words(self, sentence, words):
        bag = [0 for _ in range(len(words))]

        s_words = nltk.word_tokenize(sentence)
        s_words = [stemmer.stem(word.lower()) for word in s_words]

        for se in s_words:
            for i, w in enumerate(words):
                if w == se:
                    bag[i] = (1)

        return numpy.array(bag)

    def kelvinToCelsiusAndFahreneit(self, kelvin):
        celsius = round(kelvin - 273.15)
        fahreneit = round(celsius * (9/5) + 32)
        return celsius, fahreneit

    def hours(self):
        hour, min, sec = datetime.datetime.now().time().strftime("%H:%M:%S").split(":")
        return f'{random.choice(self.responses)} {hour} : {min} e {sec} secondi.'

    def weather(self, input):
        city = input.split(" a ")

        if len(city) > 1:
            city = city[1]
        else:
            city = "Lamezia Terme"

        base_url = "http://api.openweathermap.org/data/2.5/weather?"

        api_key = "3f9242e056658c4d2548aafbde5b6219"

        url = base_url + "appid=" + api_key + "&q=" + city

        response = requests.get(url).json()

        if response["cod"] == "404":
            return "Non ho trovato una cittá con questo nome!"

        temp_kelvin = response["main"]["temp"]
        temp_celsius, temp_fahreneit = self.kelvinToCelsiusAndFahreneit(temp_kelvin)

        weather_description = response["weather"][0]["description"]

        return f'{random.choice(self.responses)} {city} ci sono {temp_celsius} °C con {self.translator.translate(weather_description).lower()}'

    def date(self):
        day, dayn, month, yearn = datetime.date.today().strftime("%A %d %B %Y").split(" ")
        return f'{random.choice(self.responses)} {self.translator.translate(day)} {dayn} {self.translator.translate(month)} {yearn}'

    def chat(self, input):
        results = self.model.predict([self.bag_of_words(input, self.parole)])
        results_index = numpy.argmax(results)
        tag = self.labels[results_index]

        for tg in self.data["intents"]:
            if tg['tag'] == tag:
                self.responses = tg['responses']
                self.currentTag = tag

        match self.currentTag:
            case "date":
                return self.date()
            case "hours":
                return self.hours()
            case "weather":
                return self.weather(input)
            case _:
                return random.choice(self.responses)