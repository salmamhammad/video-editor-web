# import asyncio
# import websockets

# async def test_ws_connection():
#     uri = "ws://backend:8765"
#     async with websockets.connect(uri) as websocket:
#         await websocket.send("ping")
#         response = await websocket.recv()
#         assert response == "pong" 


import pytest
import asyncio
import numpy as np
import json
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch

from server import AudioVideoProcessor

# Required for testing async functions
pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_processor():
    """Fixture for mocking AudioVideoProcessor with dummy methods."""
    processor = AudioVideoProcessor(model_path="dummy_path")
    processor.device = "cpu"
    processor.model = mock.MagicMock()
    processor.whisper_model = MagicMock()
    processor.whisper_model.transcribe = MagicMock(return_value={'text': 'Hello world'})
    processor.read_audio = MagicMock(return_value=(np.ones(16000), 16000))
    processor.denoise_audio = MagicMock(return_value=np.ones(16000))
    processor.silence_removal_vad = MagicMock(return_value=[(0, 1)])
    processor.create_lstm_model = MagicMock()
    processor.create_lstm_model.return_value = MagicMock(
        fit=MagicMock(),
        predict=MagicMock(return_value=np.random.rand(1, 1))
    )
    processor.translate_text = MagicMock(return_value="translated text")
    return processor


async def test_read_audio(mock_processor):
    y, sr = mock_processor.read_audio("dummy.wav")
    assert sr == 16000
    assert isinstance(y, np.ndarray)


def test_normalize_audio(mock_processor):
    waveform = np.array([0.2, -0.4, 0.5])
    norm = mock_processor.normalize_audio(waveform)
    assert np.max(np.abs(norm)) == 1.0


def test_adjust_gain(mock_processor):
    waveform = np.ones(1000)
    adjusted = mock_processor.adjust_gain(waveform, target_dB=-10.0)
    assert not np.array_equal(waveform, adjusted)


@patch("server.sf.write")
def test_save_audio(mock_write, mock_processor):
    waveform = np.random.rand(16000)
    mock_processor.save_audio("output.wav", waveform)
    mock_write.assert_called_once()


@patch("server.cv2.VideoCapture")
@patch("server.ImageDraw.Draw")
@patch("server.ImageFont.truetype")
async def test_add_subtitles_to_video(mock_font, mock_draw, mock_cv2, mock_processor):
    cap = MagicMock()
    cap.isOpened.side_effect = [True, False]  # One iteration
    cap.read.return_value = (True, np.zeros((100, 100, 3), dtype=np.uint8))
    cap.get.side_effect = [30, 100, 100]  # fps, width, height
    mock_cv2.return_value = cap

    subtitles = [{"start_time": 0, "end_time": 2, "speaker": 0, "transcription": "Hi"}]
    websocket = AsyncMock()
    await mock_processor.add_subtitles_to_video(websocket, "client123", "vid.mp4", subtitles, "out.mp4", {})
    websocket.send.assert_called()


@patch("server.AudioSegment.from_file")
async def test_process_video(mock_audio_seg, mock_processor):
    audio_mock = MagicMock()
    audio_mock.set_frame_rate.return_value.set_channels.return_value.export.return_value = None
    mock_audio_seg.return_value = audio_mock

    websocket = AsyncMock()
    reqdata = json.dumps({
        "inputFile": "input.mp4",
        "outputFile": "out.mp4",
        "clusters": 1,
        "src": "en"
    })

    from server import process_video
    await process_video(websocket, "client123", reqdata, mock_processor)
    websocket.send.assert_called()


@patch("server.AudioSegment.from_file")
async def test_process_video_translate(mock_audio_seg, mock_processor):
    audio_mock = MagicMock()
    audio_mock.set_frame_rate.return_value.set_channels.return_value.export.return_value = None
    mock_audio_seg.return_value = audio_mock

    websocket = AsyncMock()
    reqdata = json.dumps({
        "inputFile": "input.mp4",
        "outputFile": "out.mp4",
        "clusters": 1,
        "src": "en",
        "lang": "ar"
    })

    from server import process_video_translate
    await process_video_translate(websocket, "client123", reqdata, mock_processor)
    websocket.send.assert_called()


# More test cases would follow a similar structure:
# - Mock file I/O
# - Replace heavy models (Whisper, Keras, Torch) with mocks
# - Patch external libs (cv2, ffmpeg, websockets)
# - Assert expected state or method calls
