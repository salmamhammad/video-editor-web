# import asyncio
# import websockets

# async def test_ws_connection():
#     uri = "ws://backend:8765"
#     async with websockets.connect(uri) as websocket:
#         await websocket.send("ping")
#         response = await websocket.recv()
#         assert response == "pong" 

import pytest
import numpy as np
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server import AudioVideoProcessor

# Required for async tests
pytestmark = pytest.mark.asyncio


# === FIXTURE: Create processor with mocked model ===
@pytest.fixture
def mock_processor():
    with patch('server.AudioDenoiserCNN') as MockModel, \
         patch('server.whisper.load_model') as mock_whisper:

        mock_model_instance = MagicMock()
        MockModel.return_value = mock_model_instance

        mock_whisper_model = MagicMock()
        mock_whisper.return_value = mock_whisper_model

        processor = AudioVideoProcessor("output/denoiser_model4.pth")
        processor.model = MagicMock()
        processor.whisper_model = mock_whisper_model
        return processor


# === TEST: normalize_audio ===
def test_normalize_audio(mock_processor):
    waveform = np.array([0.1, -0.5, 0.7])
    normalized = mock_processor.normalize_audio(waveform)
    assert np.isclose(np.max(np.abs(normalized)), 1.0)


# === TEST: adjust_gain ===
def test_adjust_gain(mock_processor):
    waveform = np.ones(10) * 0.1
    adjusted = mock_processor.adjust_gain(waveform, target_dB=-20)
    expected_rms = np.sqrt(np.mean(adjusted ** 2))
    target_rms = 10 ** (-20 / 20)
    assert np.isclose(expected_rms, target_rms, atol=1e-3)


# === TEST: read_audio with mocking ===
@patch('server.librosa.load')
def test_read_audio(mock_load, mock_processor):
    mock_load.return_value = (np.ones(16000), 16000)
    y, sr = mock_processor.read_audio("mock.wav")
    assert len(y) == 16000
    assert sr == 16000


# === TEST: save_audio ===
@patch('server.sf.write')
def test_save_audio(mock_write, mock_processor):
    waveform = np.random.randn(16000)
    mock_processor.save_audio("output.wav", waveform)
    mock_write.assert_called_once()


# === TEST: add_subtitles_to_video with mocks ===
@patch('server.cv2.VideoCapture')
@patch('server.cv2.VideoWriter')
@patch('server.ImageFont.truetype')
@patch('server.ImageDraw.Draw')
@patch('server.ffmpeg')
@patch('server.Image.fromarray')
@patch('server.cv2.cvtColor')
async def test_add_subtitles_to_video(
    mock_cvtColor, mock_fromarray, mock_ffmpeg, mock_draw, mock_font,
    mock_writer, mock_capture, mock_processor
):
    mock_cap_instance = MagicMock()
    mock_capture.return_value = mock_cap_instance
    mock_cap_instance.isOpened.side_effect = [True, False]
    mock_cap_instance.read.return_value = (True, np.zeros((720, 1280, 3)))

    websocket = AsyncMock()
    subtitles = [{
        'start_time': 0, 'end_time': 2, 'speaker': 1, 'transcription': "Hello"
    }]
    await mock_processor.add_subtitles_to_video(websocket, "test_id", "in.mp4", subtitles, "out.mp4", {})

    assert websocket.send.called
    mock_writer.return_value.write.assert_called()

