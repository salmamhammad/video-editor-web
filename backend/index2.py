import os
import numpy as np
import librosa
import soundfile as sf
import cv2
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
import whisper
import webrtcvad
import subprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Flatten
from tensorflow.keras.optimizers import Adam
from pydub import AudioSegment
import textwrap
import torch
import torchaudio
from audioDenoiserCNN  import AudioDenoiserCNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import arabic_reshaper
from bidi.algorithm import get_display
from transformers import MarianMTModel, MarianTokenizer
import asyncio
import websockets
import json
import time

class AudioVideoProcessor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AudioDenoiserCNN().to(self.device) 
        checkpoint = torch.load(model_path, map_location=self.device)  
        # Load model to correct device
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device) 
        # make the model is on the correct device
        self.model.eval()
        # make Whisper runs on the same device
        self.whisper_model = whisper.load_model("medium", device=self.device)

        # Load pre-downloaded translation models
        self.model_ar_en, self.tokenizer_ar_en = self.load_model_from_disk('ar', 'en')
        self.model_ar_ru, self.tokenizer_ar_ru = self.load_model_from_disk('ar', 'ru')
        self.model_en_ar, self.tokenizer_en_ar = self.load_model_from_disk('en', 'ar')
        self.model_en_ru, self.tokenizer_en_ru = self.load_model_from_disk('en', 'ru')
        self.model_ru_en, self.tokenizer_ru_en = self.load_model_from_disk('ru', 'en')
        self.model_ru_ar, self.tokenizer_ru_ar = self.load_model_from_disk('ru', 'ar')

    def load_model_from_disk(self, src, tgt):
        model_path = f"models/mt-{src}-{tgt}"   
        # Check if the model directory exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        # Force loading from local files only
        model = MarianMTModel.from_pretrained(model_path, local_files_only=True)
        tokenizer = MarianTokenizer.from_pretrained(model_path, local_files_only=True)
        return model, tokenizer
        
    def read_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=16000)
        return y, sr

    # Function to create LSTM model
    def create_lstm_model(self, input_shape):
      model = Sequential()
      model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
      model.add(Dropout(0.2))
      model.add(LSTM(64))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(1, activation='linear'))  
      model.compile(optimizer='adam', loss='mse')
      return model

    def silence_removal_vad(self, y, sr, frame_duration_ms=30):
        vad = webrtcvad.Vad(2)
        y = (y * 32767).astype(np.int16)
        frame_size = int(sr * frame_duration_ms / 1000)
        if frame_size % 2 != 0:
            frame_size += 1
        padding = frame_size - (len(y) % frame_size)
        y = np.pad(y, (0, padding), mode="constant")
        frames = [y[i: i + frame_size] for i in range(0, len(y), frame_size)]
        speech_segments = []
        start_time = None
        for i, frame in enumerate(frames):
            frame_bytes = frame.tobytes()
            is_speech = vad.is_speech(frame_bytes, sample_rate=sr)
            current_time = i * frame_duration_ms / 1000.0
            if is_speech:
                if start_time is None:
                    start_time = current_time
            elif start_time is not None:
                end_time = current_time
                speech_segments.append((start_time, end_time))
                start_time = None
        if start_time is not None:
            speech_segments.append((start_time, len(y) / sr))
        return speech_segments
    # Normalize the audio
    def normalize_audio(self,waveform):
        return waveform / np.max(np.abs(waveform))
    # Apply a low-pass filter
    def apply_low_pass_filter(self,waveform, sr, cutoff=5000):
        nyquist = sr / 2
        sos = scipy.signal.butter(10, cutoff / nyquist, btype="low", output="sos")
        return  scipy.signal.sosfilt(sos, waveform)

    # Adjust gain
    def adjust_gain(self,waveform, target_dB=-20.0):
        rms = np.sqrt(np.mean(waveform ** 2))
        scalar = 10 ** (target_dB / 20) / rms
        return waveform * scalar
   
    # Denoise the audio
    def denoise_audio(self, noisy_waveform):
        noisy_waveform = torch.tensor(noisy_waveform, dtype=torch.float32, device=self.device).unsqueeze(0)  # Move input to GPU/CPU
        with torch.no_grad():
            denoised_waveform = self.model(noisy_waveform)
        return denoised_waveform.squeeze().cpu().numpy() 

    async  def extract_speech_segments_and_cluster(self, websocket, input_audio, output_dir, n_clusters=2,language='en'):
        os.makedirs(output_dir, exist_ok=True)
        # print(f"extract_speech_segments_and_cluster 1 ")
         # Step 1: Read and Denoise Audio
        # await websocket.send(json.dumps({"status": "Reading & Denoising Audio", "progress": 40}))
        noisy_waveform, sr = self.read_audio(input_audio)
        # noisy_waveform = self.normalize_audio(noisy_waveform)
        denoised_waveform = self.denoise_audio(noisy_waveform)
        # denoised_waveform = self.apply_low_pass_filter(denoised_waveform, sr)
        # denoised_waveform = self.adjust_gain(denoised_waveform)
        # Step 2: Silence Removal
        # await websocket.send(json.dumps({"status": "Removing Silence", "progress": 45}))
        merged_segments = self.silence_removal_vad(denoised_waveform, sr)
        # print(f"extract_speech_segments_and_cluster 2 ")

        all_mfcc_features = []
        segment_info = []
        transcriptions = []

        for i, (start, end) in enumerate(merged_segments):
            # await websocket.send(json.dumps({"status": f"Processing Segment {i+1}", "progress": 50 + (i * 5)}))
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            speech_segment = denoised_waveform[start_sample:end_sample]

            # Save segment
            temp_filename = os.path.join(output_dir, f"segment_{i + 1}.wav")
            sf.write(temp_filename, speech_segment, sr)

            # Transcribe using Whisper
            transcription_result = self.whisper_model.transcribe(temp_filename, language=language)
            transcriptions.append(transcription_result['text'])
            segment_info.append((start, end))

            # Extract MFCCs for Clustering
            mfcc = librosa.feature.mfcc(y=speech_segment, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)  
            all_mfcc_features.append(mfcc_mean)

           # Step 4: Perform Speaker Clustering
        # await websocket.send(json.dumps({"status": "Clustering Speakers", "progress": 60}))
        # Convert all MFCC features to a numpy array
        all_mfcc_features = np.array(all_mfcc_features)
        # print(f"extract_speech_segments_and_cluster3 ")

        # Reshape MFCC features for LSTM input (samples, time steps, features)
        all_mfcc_features_reshaped = all_mfcc_features.reshape((all_mfcc_features.shape[0], 1, all_mfcc_features.shape[1]))

        # Create and train the LSTM model on MFCC features
        lstm_model = self.create_lstm_model((all_mfcc_features_reshaped.shape[1], all_mfcc_features_reshaped.shape[2]))
        lstm_model.fit(all_mfcc_features_reshaped, np.zeros(len(all_mfcc_features_reshaped)), epochs=10, batch_size=32)

        # Extract embeddings from the LSTM model
        reduced_features = lstm_model.predict(all_mfcc_features_reshaped)
        # print(f"extract_speech_segments_and_cluster 4 ")

        if len(reduced_features) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(reduced_features)
        else:
            cluster_labels = [0] * len(segment_info)  # Default to single speaker if not enough data

        predicted_segments = []
        for i, (start, end) in enumerate(segment_info):
            predicted_segments.append({
                "start_time": start,
                "end_time": end,
                "speaker": cluster_labels[i],
                "transcription": transcriptions[i]
            })
            
        # Print the clustered segments with start_time, end_time, speaker, and transcription
        for segment in predicted_segments:
            print(f"Start Time: {segment['start_time']:.2f}s, End Time: {segment['end_time']:.2f}s, "
              f"Speaker: {segment['speaker']}, Transcription: {segment['transcription']}")

 
        return predicted_segments, cluster_labels


    async def add_subtitles_to_video(self,websocket, video_path, subtitles, final_output_path):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = "temp_video.mp4"
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        frame_no = 0
        font = ImageFont.truetype("arial.ttf", 15)
        # print(f"add_subtitles_to_video 1 ")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # await websocket.send(json.dumps({"status": f"Adding Subtitle {frame_no}", "progress": 80 + int((frame_no / fps) * 20)}))
            time_sec = frame_no / fps
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            for sub in subtitles:
                if sub['start_time'] <= time_sec <= sub['end_time']:
                    text = f"Speaker {sub['speaker']} :  {sub['transcription']}"
                    wrapped_text = textwrap.fill(text, width=40)
                    x, y = 50, height - 120
                    for line in wrapped_text.split("\n"):
                        draw.text((x, y), line, font=font, fill=(255, 255, 255))
                        y += 25
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            out.write(frame)
            frame_no += 1

        cap.release()
        out.release()
        ffmpeg_command = [
           "ffmpeg", "-i", temp_video_path, "-c:v", "libx264", "-crf", "23", "-preset", "fast",
           "-an", final_output_path, "-y"
         ]
    
        # print(f"add_subtitles_to_video 2 ")
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # print(f"✅ Video processed and saved as: {final_output_path}")
        # await websocket.send(json.dumps({"status": "Video Processing Complete", "progress": 100}))
        
            # Function to load the Marian NMT translation model
  
    def load_model(self, src_lang, tgt_lang):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to(self.device)
        return model, tokenizer

    def translate_text(self, text, src_lang, tgt_lang):
        # Check language pair and use the corresponding model/tokenizer
        if src_lang == "ar" and tgt_lang == "en":
            inputs = self.tokenizer_ar_en(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.model_ar_en.generate(**inputs)
            return self.tokenizer_ar_en.decode(translated[0], skip_special_tokens=True)
    
        elif src_lang == "en" and tgt_lang == "ar":
            inputs = self.tokenizer_en_ar(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.model_en_ar.generate(**inputs)
            return self.tokenizer_en_ar.decode(translated[0], skip_special_tokens=True)
    
        elif src_lang == "ru" and tgt_lang == "en":
            inputs = self.tokenizer_ru_en(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.model_ru_en.generate(**inputs)
            return self.tokenizer_ru_en.decode(translated[0], skip_special_tokens=True)
    
        elif src_lang == "en" and tgt_lang == "ru":
            inputs = self.tokenizer_en_ru(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.model_en_ru.generate(**inputs)
            return self.tokenizer_en_ru.decode(translated[0], skip_special_tokens=True)
    
        elif src_lang == "ar" and tgt_lang == "ru":
            inputs = self.tokenizer_ar_ru(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.model_ar_ru.generate(**inputs)
            return self.tokenizer_ar_ru.decode(translated[0], skip_special_tokens=True)
    
        elif src_lang == "ru" and tgt_lang == "ar":
            inputs = self.tokenizer_ru_ar(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            translated = self.model_ru_ar.generate(**inputs)
            return self.tokenizer_ru_ar.decode(translated[0], skip_special_tokens=True)
    
        else:
            raise ValueError(f"Unsupported translation direction: {src_lang} to {tgt_lang}")

    async def add_subtitles_to_video_translate(self,websocket, video_path, subtitles, final_output_path,src,lang):
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_video_path = "temp_video.mp4"
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
        frame_no = 0
        font = ImageFont.truetype("arial.ttf", 15)
        
        for sub in subtitles:
             translatetext=self.translate_text( sub['transcription'], src, lang)
             print(f"add_subtitles_to_video : {translatetext}")
             sub['translatetext']=translatetext

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # await websocket.send(json.dumps({"status": f"Adding Subtitle {frame_no}", "progress": 80 + int((frame_no / fps) * 20)}))
            time_sec = frame_no / fps
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            for sub in subtitles:
                if sub['start_time'] <= time_sec <= sub['end_time']:
                    text = f"Speaker {sub['speaker']} :  {sub['translatetext']}"
                    wrapped_text = textwrap.fill(text, width=40)
                    x, y = 50, height - 120
                    for line in wrapped_text.split("\n"):
                        draw.text((x, y), line, font=font, fill=(255, 255, 255))
                        y += 25
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            out.write(frame)
            frame_no += 1

        cap.release()
        out.release()
        ffmpeg_command = [
           "ffmpeg", "-i", temp_video_path, "-c:v", "libx264", "-crf", "23", "-preset", "fast",
           "-an", final_output_path, "-y"
         ]
    
        # print(f"add_subtitles_to_video 2 ")
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  
    def save_audio(self,output_path, waveform, sample_rate=16000):
        sf.write(output_path, waveform, sample_rate)

# model_path = "output/denoiser_model4.pth"
# processor = AudioVideoProcessor(model_path)
# video_path = "dd.mp4"
# input_lang = "en"
# output_lang = "ar"

# audio_output_path = "output_audio.wav"
# final_output_path = "final_video_with_audio2.mp4"
# AudioSegment.from_file(video_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")
# predicted_segments, _ = processor.extract_speech_segments_and_cluster(audio_output_path, "output_segments", n_clusters=3,language=input_lang)
# processor.add_subtitles_to_video(video_path, predicted_segments, final_output_path)
