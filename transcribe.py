#!/usr/bin/env python
#
# Copyright 2016 IBM
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may obtain
# a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import argparse
import base64
import configparser
import json
import threading
import time
import sys

import pyaudio
import websocket
from websocket._abnf import ABNF
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import QThread, pyqtSignal
import openai

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
FINALS = []
LAST = None

REGION_MAP = {
    'us-east': 'api.us-east.speech-to-text.watson.cloud.ibm.com/instances/170112cf-ba4e-4e08-953d-3bb8ad6faa97',
    'us-south': 'stream.watsonplatform.net',
    'eu-gb': 'stream.watsonplatform.net',
    'eu-de': 'stream-fra.watsonplatform.net',
    'au-syd': 'gateway-syd.watsonplatform.net',
    'jp-tok': 'gateway-syd.watsonplatform.net',
}
openai.api_key = 'your open-ai key'
class SpeechThread(QThread):
    new_message = pyqtSignal(str)
    finished = pyqtSignal(str)
    sentiment_result = pyqtSignal(str)  # Signal to emit sentiment results

    def __init__(self, timeout):
        super().__init__()
        self.timeout = timeout
        self.transcript_chunk = ""  # Store the transcript for analysis

    def run(self):
        ws = websocket.WebSocketApp(url,
                                    header=headers,
                                    on_message=self.on_message,
                                    on_error=self.on_error,
                                    on_close=self.on_close)
        ws.on_open = lambda ws: self.on_open(ws)
        ws.run_forever()

    def on_message(self, ws, msg):
        """Handle incoming messages."""
        global LAST
        data = json.loads(msg)
        if "results" in data:
            partial_transcript = data['results'][0]['alternatives'][0]['transcript'].strip()
            
            if data["results"][0]["final"]:
                FINALS.append(data)
                LAST = None
                self.transcript_chunk += " " + partial_transcript
                self.transcript_chunk = self.transcript_chunk.strip()

                # Emit the full transcript
                transcript = "".join([x['results'][0]['alternatives'][0]['transcript']
                                      for x in FINALS])
                self.new_message.emit(transcript)

                # Analyze the sentiment of the entire chunk
                self.analyze_sentiment(self.transcript_chunk)
            else:
                LAST = data
                self.new_message.emit(self.transcript_chunk + " " + partial_transcript)

    def on_error(self, ws, error):
        """Handle errors."""
        print(error)

    def on_close(self, ws, close_status_code, close_msg):
        """Handle the closing of the WebSocket connection."""
        global LAST
        if LAST:
            FINALS.append(LAST)
        transcript = "".join([x['results'][0]['alternatives'][0]['transcript']
                              for x in FINALS])
        self.finished.emit(transcript)

    def on_open(self, ws):
        """Handle WebSocket opening."""
        data = {
            "action": "start",
            "content-type": "audio/l16;rate=%d" % RATE,
            "continuous": True,
            "interim_results": True,
            "word_confidence": True,
            "timestamps": True,
            "max_alternatives": 3
        }
        ws.send(json.dumps(data).encode('utf8'))
        threading.Thread(target=read_audio, args=(ws, self.timeout)).start()

    def analyze_sentiment(self, text):
        """Send the text to OpenAI for sentiment analysis."""
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Analyze the sentiment of the following text and respond with one of three answers: NEUTRAL, NEGATIVE, POSITIVE."},
                      {"role": "user", "content": text}]
        )
        sentiment = response.choices[0].message.content.strip()
        self.sentiment_result.emit(sentiment)

def read_audio(ws, timeout):
    """Read audio and send it to the WebSocket."""
    global RATE
    p = pyaudio.PyAudio()
    RATE = int(p.get_default_input_device_info()['defaultSampleRate'])
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")
    rec = timeout or RECORD_SECONDS

    for i in range(0, int(RATE / CHUNK * rec)):
        data = stream.read(CHUNK)
        ws.send(data, ABNF.OPCODE_BINARY)

    # Disconnect the audio stream
    stream.stop_stream()
    stream.close()
    print("* done recording")

    # Send stop message to get the final response from STT
    data = {"action": "stop"}
    ws.send(json.dumps(data).encode('utf8'))
    time.sleep(1)
    ws.close()

    # Terminate the audio device
    p.terminate()

def get_url():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    region = config.get('auth', 'region')
    host = REGION_MAP[region]
    return ("wss://{}/v1/recognize"
           "?model=en-AU_BroadbandModel").format(host)

def get_auth():
    config = configparser.RawConfigParser()
    config.read('speech.cfg')
    apikey = config.get('auth', 'apikey')
    return ("apikey", apikey)

def parse_args():
    parser = argparse.ArgumentParser(description='Transcribe Watson text in real time')
    parser.add_argument('-t', '--timeout', type=int, default=5)
    args = parser.parse_args()
    return args

def create_window():
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Watson Speech to Text with Sentiment Analysis")
    window.setGeometry(100, 100, 800, 600)

    widget = QWidget()
    layout = QVBoxLayout()

    text_edit = QTextEdit()
    text_edit.setReadOnly(True)

    sentiment_label = QLabel("Sentiment: ")
    layout.addWidget(text_edit)
    layout.addWidget(sentiment_label)

    widget.setLayout(layout)
    window.setCentralWidget(widget)

    def update_text(message):
        text_edit.setText(message)  # Replace the entire text

    def update_sentiment(sentiment):
        sentiment_label.setText("Sentiment: " + sentiment)  # Update sentiment label

    def update_transcript(transcript):
        text_edit.setText("\nClosing transcript:\n" + transcript)  # Replace the entire text

    args = parse_args()
    global url, headers
    headers = {}
    userpass = ":".join(get_auth())
    headers["Authorization"] = "Basic " + base64.b64encode(userpass.encode()).decode()
    url = get_url()

    speech_thread = SpeechThread(args.timeout)
    speech_thread.new_message.connect(update_text)
    speech_thread.finished.connect(update_transcript)
    speech_thread.sentiment_result.connect(update_sentiment)
    speech_thread.start()

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    create_window()
