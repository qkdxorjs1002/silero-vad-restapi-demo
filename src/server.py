import uuid
from flask_api import status
from flask import Flask, request
from IPython.display import Audio
import torch
import json

# Init silero-vad
SAMPLING_RATE = 16000

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


# Demo REST API
app = Flask(__name__)


@app.route('/vad_timestamp', methods=['GET'])
def getVadTimestamp():
    if request.method == 'GET':
        filename = makeUuid()
        url = request.args.get("url")
        torch.hub.download_url_to_file(url, "./" + filename)
        result = json.dumps(vadTimestamp(filename))

    return result, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}


@app.route('/vad_timestamp', methods=['POST'])
def postVadTimestamp():
    if request.method == 'POST':
        filename = makeUuid()
        with open("./" + filename, "wb") as file:
            file.write(request.data)
        result = json.dumps(vadTimestamp(filename))

    return result, status.HTTP_200_OK, {"Content-Type": "application/json; charset=utf-8", "Access-Control-Allow-Origin": "*"}


def vadTimestamp(filename):
    wav = read_audio(filename, sampling_rate=SAMPLING_RATE)
    result = get_speech_timestamps(wav, model, threshold=0.65, sampling_rate=SAMPLING_RATE, return_seconds=True)

    return result


def makeUuid():
    return str(uuid.uuid1())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
