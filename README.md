# silero-vad-restapi-demo

A demo project to test [silero-vad](https://github.com/snakers4/silero-vad) using REST API

## Requirements

```text
wheel
numpy
ipython
torch >= 1.9.0
torchaudio >= 0.9.0
flask
flask_api
```

## Usage

```bash
pip install -r requirements.txt
python ./src/server.py
```

## REST API

### GET ```/vad_timestamp```

> request vad timestamp from wav file url

Request

```text
http://localhost:8080/vad_timestamp?url=http://example.com/test.wav
```

| Query | Type   | Description  |
| ----- | ------ | ------------ |
| url   | string | Wav file URL |

Response

```JSON
[
  {
    "start": 1.9,
    "end": 3.4
  }
]
```

| Response | Type      | Description                      |
| -------- | --------- | -------------------------------- |
| body     | JSONArray | timestamp list of voice activity |

### POST ```/vad_timestamp```

> request vad timestamp from wav file

Request

```text
http://localhost:8080/vad_timestamp
```

| Body | Type  | Description  |
| ---- | ----- | ------------ |
| data | bytes | Audio Binary |

Response

```JSON
[
  {
    "start": 1.9,
    "end": 3.4
  }
]
```

| Response | Type      | Description                      |
| -------- | --------- | -------------------------------- |
| body     | JSONArray | timestamp list of voice activity |