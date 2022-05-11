from time import time
import numpy
import torch
import os
import json
import librosa, librosa.display 
import matplotlib.pyplot as plot

FIG_SIZE = (15, 12)

SAMPLING_RATE = 16000

WAV_PATH = "/Users/paragonnov/Downloads/VAD_ACC_TEST_SAMPLE/other/wav"
JSON_PATH = "/Users/paragonnov/Downloads/VAD_ACC_TEST_SAMPLE/other/json"

torch.set_num_threads(1)

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils

for dir, subdirs, files in os.walk(WAV_PATH):
    for file in files:
        if file.find(".wav") > 0:
            plot.interactive(False)
            fig, ax = plot.subplots(figsize=FIG_SIZE, nrows=2, ncols=1, sharex=True)

            filePath = dir + "/" + file
            sample, sampleRate = librosa.load(filePath, 16000)

            wav = read_audio(filePath, sampling_rate=SAMPLING_RATE)
            speechTimestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)
            speechTimestampsLabel = []
            for timestamp in speechTimestamps:
                speechTimestampsLabel.append({
                    "start": timestamp["start"] / 16000,
                    "end": timestamp["end"] / 16000
                })

            markers = numpy.empty(sample.size)
            markers.fill(0)
            for mark in speechTimestamps:
                for idx in range(mark["start"], mark["end"]):
                    markers[idx] = 1

            jsonPath = filePath.replace(WAV_PATH, JSON_PATH).replace(".wav", ".json")
            annotation = json.load(open(jsonPath, 'r'))
            
            annotationSpeechTime = annotation.get("Miscellaneous_Info", None)
            if annotationSpeechTime == None:
                annotationOther = annotation.get("Other")
                if annotationOther.get("SpeechStart", None) != None: 
                    annotationSpeechTime = annotationOther

            if annotationSpeechTime != None:
                annotationSpeechTimeLabel = {
                    "start": annotationSpeechTime["SpeechStart"],
                    "end": annotationSpeechTime["SpeechEnd"]
                }
                jsonSpeechStart = int(float(annotationSpeechTime["SpeechStart"]) * sampleRate)
                jsonSpeechEnd = int(float(annotationSpeechTime["SpeechEnd"]) * sampleRate)
            
                jsonMarkers = numpy.empty(sample.size)
                jsonMarkers.fill(0)
                for idx in range(jsonSpeechStart, jsonSpeechEnd):
                    jsonMarkers[idx] = 1
            
            librosa.display.waveshow(sample, sr=sampleRate, alpha=0.65, color='b', ax=ax[0], offset=0.0)
            librosa.display.specshow(librosa.amplitude_to_db(numpy.abs(librosa.stft(sample)), ref=numpy.max), sr=sampleRate, y_axis='linear', x_axis='time', ax=ax[1])
            librosa.display.waveshow(markers, sr=sampleRate, alpha=0.2, color='g', ax=ax[0], offset=0.0, label="silero - " + str(speechTimestampsLabel))
            if annotationSpeechTimeLabel != None:
                librosa.display.waveshow(jsonMarkers, sampleRate, alpha=0.2, color='r', ax=ax[0], offset=0.0, label="speechfinder - " + str(annotationSpeechTimeLabel))
            ax[0].legend()
            ax[0].set_title("Harmonic Waveform")
            ax[0].label_outer()
            ax[1].set_title("Linear Spectrum")
            fig.suptitle("file: " + filePath, size="x-large")
            if not os.path.exists("./out"): os.makedirs("./out")
            plot.savefig("out/" + file + ".png")
            plot.close('all')

