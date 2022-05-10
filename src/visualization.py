import numpy
import torch
import os
import json
import librosa, librosa.display 
import matplotlib.pyplot as plot

FIG_SIZE = (15,10)

SAMPLING_RATE = 16000

WAV_PATH = "/Users/paragonnov/Downloads/VAD_ACCURATE_TEST_SAMPLE"
JSON_PATH = "/Users/paragonnov/Downloads/VAD_ACCURATE_TEST_SAMPLE"

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
            filePath = dir + "/" + file
            sample, sampleRate = librosa.load(filePath, 16000)

            wav = read_audio(filePath, sampling_rate=SAMPLING_RATE)
            speech_timestamps = get_speech_timestamps(wav, model, threshold=0.5, sampling_rate=SAMPLING_RATE)

            markers = numpy.empty(sample.size)
            markers.fill(0)
            for mark in speech_timestamps:
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
                jsonSpeechStart = int(float(annotationSpeechTime["SpeechStart"]) * sampleRate)
                jsonSpeechEnd = int(float(annotationSpeechTime["SpeechEnd"]) * sampleRate)
            
                jsonMarkers = numpy.empty(sample.size)
                jsonMarkers.fill(0)
                for idx in range(jsonSpeechStart, jsonSpeechEnd):
                    jsonMarkers[idx] = 1
            
            plot.interactive(False)
            plot.figure(figsize=FIG_SIZE)
            librosa.display.waveshow(sample, sampleRate, alpha=0.65, color='b')
            librosa.display.waveshow(markers, sampleRate, alpha=0.2, color='g')
            if annotationSpeechTime != None:
                librosa.display.waveshow(jsonMarkers, sampleRate, alpha=0.2, color='r')
            plot.xlabel("Sample")
            plot.ylabel("Amplitude")
            plot.title("file: " + filePath + "\nsilero: " + str(speech_timestamps) + "\nSpeechFinder: " + str(annotationSpeechTime))
            if not os.path.exists("./out"): os.makedirs("./out")
            plot.savefig("out/" + file + ".png")
            plot.close('all')

