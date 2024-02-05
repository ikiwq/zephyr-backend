# Zephir
Welcome to Zephyr! This repository contains the backend codebase for the Zephyr project. You can find the frontend codebase [here](https://github.com/ikiwq/zephyr-frontend).

## Project Overview

Zephyr is designed to emulate a personal assistant using TensorFlow. The goal is to create a customizable personal assistant.
### Built with
[![My Skills](https://skillicons.dev/icons?i=python,flask,tensorflow&theme=light)](https://skillicons.dev)

## Requisites
- Python 3.11 or higher. Download python from the official [website](https://www.python.org)
- Pip dependencies. Install with ```pip install``` the following libraries:
    - numpy
    - tflearn
    - pickle
    - flask
    - translate
       
## Getting Started

To get started with the Zephyr frontend, follow these steps:
Clone this repository to your local machine:

    git clone https://github.com/ikiwq/zephyr-backend.git

And run the main file with python:

    python main.py

## Usage
Inside the intents.json, there is a list of behaviors. Each behavior has a tag, a list of patterns (that represent a hypotethical user input), and a list of possible responses.
At the end of the zephyr.py file, there is a switch of possible tags associated with a function; otherwise, a random answer related to the tag will be returned.

## License

This project is licensed under the MIT License.
