import asyncio
import websockets
import json
import functools  # ✅ Import functools for partial function
from pydub import AudioSegment
import sys
import ffmpeg
import os
import time
import gc
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
import cv2
import numpy as np
import openshot 
os.environ["OMP_NUM_THREADS"] = "2"  # Limits OpenMP to 2 threads  
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # Limits OpenBLAS (used by OpenShot)
sys.stdout.reconfigure(encoding='utf-8')
connected_clients = {}
UPLOAD_FOLDER = "/app/backend/"
client_projects = {}  # 🔹 Stores projects per client
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
        self.model_ar_en, self.tokenizer_ar_en = self.load_model('ar', 'en')
        self.model_ar_ru, self.tokenizer_ar_ru = self.load_model('ar', 'ru')
        self.model_en_ar, self.tokenizer_en_ar = self.load_model('en', 'ar')
        self.model_en_ru, self.tokenizer_en_ru = self.load_model('en', 'ru')
        self.model_ru_en, self.tokenizer_ru_en = self.load_model('ru', 'en')
        self.model_ru_ar, self.tokenizer_ru_ar = self.load_model('ru', 'ar')

     # Function to load the Marian NMT translation model
    def load_model(self,src_lang, tgt_lang):
        model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        model.to(self.device)

        return model, tokenizer

        
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

    async  def extract_speech_segments_and_cluster(self, websocket,client_id, input_audio, output_dir, n_clusters=2,language='en'):
        os.makedirs(output_dir, exist_ok=True)
        # print(f"extract_speech_segments_and_cluster 1 ")
         # Step 1: Read and Denoise Audio
        response={"status": "Reading & Denoising Audio", "progress": 16, "time_left": 0,'reqdatajson':{}}
        await websocket.send(json.dumps({"clientId": client_id,'response':response}))
        noisy_waveform, sr = self.read_audio(input_audio)
        # noisy_waveform = self.normalize_audio(noisy_waveform)
        denoised_waveform = self.denoise_audio(noisy_waveform)
        # denoised_waveform = self.apply_low_pass_filter(denoised_waveform, sr)
        # denoised_waveform = self.adjust_gain(denoised_waveform)
        # Step 2: Silence Removal
        response={"status": "Removing Silence", "progress": 20, "time_left": 0,'reqdatajson':{}}
        await websocket.send(json.dumps({"clientId": client_id,'response':response}))
        merged_segments = self.silence_removal_vad(denoised_waveform, sr)
        # print(f"extract_speech_segments_and_cluster 2 ")

        all_mfcc_features = []
        segment_info = []
        transcriptions = []
        total=len(merged_segments);
        subtime=total/200
        for i, (start, end) in enumerate(merged_segments):
            response={"status": f"Processing Segment {i+1}", "progress": 20+ (i*subtime), "time_left": 0,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))         
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
        response={"status": "Clustering Speakers", "progress": 70, "time_left": 0,'reqdatajson':{}}
        await websocket.send(json.dumps({"clientId": client_id,'response':response}))        # Convert all MFCC features to a numpy array
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


    async def add_subtitles_to_video(self,websocket,client_id,video_path, subtitles, final_output_path,reqdata):
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

            response={"status": f"Adding Subtitle {frame_no}", "progress": 62 + int((frame_no / fps) * 0.1), "time_left": 0,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response})) 
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
        response={"status": "Completed", "progress": 100,'reqdatajson':reqdata}
        await websocket.send(json.dumps({"clientId": client_id,'response':response}))
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

    async def add_subtitles_to_video_translate(self,websocket,client_id, video_path, subtitles, final_output_path,src,lang,reqdata):
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

            response={"status": f"Adding Subtitle {frame_no}", "progress": 62 + int((frame_no / fps) * 0.1), "time_left": 0,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))        
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
        response={"status": "Completed", "progress": 100,'reqdatajson':reqdata}
        await websocket.send(json.dumps({"clientId": client_id,'response':response}))
  
    def save_audio(self,output_path, waveform, sample_rate=16000):
        sf.write(output_path, waveform, sample_rate)
#################################################################################################################################
async def create_project(client_id, input_video, fps):
    """Creates a new OpenShot project for a client if not exists."""
    if client_id in client_projects:
        return client_projects[client_id]["timeline"]  # Return existing project

    reader = openshot.FFmpegReader(input_video)
    reader.Open()

    # Get the actual duration of the video
    actual_duration = reader.info.duration  # Should be 71 seconds (1:11)
    
    # Force the desired FPS
    desired_fps = openshot.Fraction(fps, 1)
   
    timeline = openshot.Timeline(
        reader.info.width, reader.info.height, desired_fps,  
        44100, 2, openshot.LAYOUT_STEREO
    )

    # Set the correct FPS in reader
    reader.info.fps = desired_fps  

    # Create the clip with correct duration
    clip = openshot.Clip(reader)
    clip.Start = 0
    clip.End = actual_duration  # Ensure it stops at 71 seconds

    timeline.AddClip(clip)
    timeline.info.duration=actual_duration
    print(f"✅ actual_duration: {actual_duration} ")
    print(f"🎥 reader.info.duration: {reader.info.duration}")
    print(f"🎥 timeline.info.duration: {timeline.info.duration}")
    print(f"🎥 Input FPS: {reader.info.fps.num}/{reader.info.fps.den}")
    print(f"🎥 Timeline FPS: {timeline.info.fps.num}/{timeline.info.fps.den}")
    client_projects[client_id] = {
        "timeline": timeline,
        "reader": reader,
        "clip": clip,
        "effects": []
    }

    print(f"✅ Created new project for Client {client_id} with {fps} FPS, duration: {actual_duration} sec")
    return timeline

async def apply_blur_filter(websocket ,client_id,reqdata):
    data = json.loads(reqdata)
    # intensity=data.get("intensity")
    input_video =os.path.join(UPLOAD_FOLDER, data.get("inputFile"))
    # output_video =os.path.join(UPLOAD_FOLDER, data.get("outputFile"))
    fps=int(data.get("fps"))
    sigma=int(data.get("sigma"))
    iterations=int(data.get("iterations"))
    horizontal_radius=int(data.get("horizontal_radius"))
    vertical_radius=int(data.get("vertical_radius"))


    timeline = await create_project(client_id, input_video, fps)
    clip = client_projects[client_id]["clip"]

    # Apply a Blur effect
    blur_effect = openshot.Blur()
    blur_effect.horizontal_radius.Value = horizontal_radius
    blur_effect.vertical_radius.Value = vertical_radius
    blur_effect.sigma.Value = sigma
    blur_effect.iterations.Value = iterations

    clip.AddEffect(blur_effect)  
    client_projects[client_id]["effects"].append(blur_effect)
    await websocket.send(json.dumps({"clientId": client_id, "response": {"status": "Blur Applied"}}))

async def apply_brightness_filter(websocket, client_id, reqdata):
    data = json.loads(reqdata)
    fps = int(data.get("fps"))
    input_video = os.path.join(UPLOAD_FOLDER, data.get("inputFile"))

    timeline = await create_project(client_id, input_video, fps)
    clip = client_projects[client_id]["clip"]

    brightness_effect = openshot.Brightness()
    brightness_effect.brightness.Value = float(data.get("brightness"))

    clip.AddEffect(brightness_effect)  
    client_projects[client_id]["effects"].append(brightness_effect)

    await websocket.send(json.dumps({"clientId": client_id, "response": {"status": "Brightness Applied"}}))


async def apply_contrast_filter(websocket, client_id, reqdata):
    data = json.loads(reqdata)
    fps = int(data.get("fps"))
    input_video = os.path.join(UPLOAD_FOLDER, data.get("inputFile"))

    timeline = await create_project(client_id, input_video, fps)
    clip = client_projects[client_id]["clip"]

    contrast_effect = openshot.Contrast()
    contrast_effect.contrast.Value = float(data.get("contrast"))

    clip.AddEffect(contrast_effect)  
    client_projects[client_id]["effects"].append(contrast_effect)

    await websocket.send(json.dumps({"clientId": client_id, "response": {"status": "Contrast Applied"}}))


async def apply_grayscale_filter(websocket, client_id, reqdata):
    data = json.loads(reqdata)
    fps = int(data.get("fps"))
    input_video = os.path.join(UPLOAD_FOLDER, data.get("inputFile"))

    timeline = await create_project(client_id, input_video, fps)
    clip = client_projects[client_id]["clip"]

    grayscale_effect = openshot.Grayscale()
    clip.AddEffect(grayscale_effect)  
    client_projects[client_id]["effects"].append(grayscale_effect)

    await websocket.send(json.dumps({"clientId": client_id, "response": {"status": "Grayscale Applied"}}))


async def apply_sepia_filter(websocket, client_id, reqdata):
    data = json.loads(reqdata)
    fps = int(data.get("fps"))
    input_video = os.path.join(UPLOAD_FOLDER, data.get("inputFile"))

    timeline = await create_project(client_id, input_video, fps)
    clip = client_projects[client_id]["clip"]

    sepia_effect = openshot.Sepia()
    clip.AddEffect(sepia_effect)  
    client_projects[client_id]["effects"].append(sepia_effect)

    await websocket.send(json.dumps({"clientId": client_id, "response": {"status": "Sepia Applied"}}))



async def render_video(websocket, client_id, reqdata):
    """Renders the final video for a client."""
    data = json.loads(reqdata)
    output_video = os.path.join(UPLOAD_FOLDER, data.get("outputFile"))
    fps = int(data.get("fps"))

    if client_id not in client_projects:
        await websocket.send(json.dumps({"clientId": client_id, "response": {"error": "No project found"}}))
        return

    timeline = client_projects[client_id]["timeline"]
    reader = client_projects[client_id]["reader"]

    # Create FFmpegWriter for output
    writer_info = openshot.WriterInfo()
    writer_info.fps = timeline.info.fps
    writer_info.width = timeline.info.width
    writer_info.height = timeline.info.height
    writer_info.video_timebase = openshot.Fraction(1, timeline.info.fps.num)
    writer_info.sample_rate = 48000
    writer_info.channels = 2
    # Initialize the FFmpegWriter
    writer = openshot.FFmpegWriter(output_video)
    writer.info = writer_info

 
    # Set video options for the output file (simplified)
    writer.SetVideoOptions(True, "libx264", writer_info.fps, writer_info.width, writer_info.height, 
                       openshot.Fraction(1, 1), False, False, 0)

    # Set audio options (disable if audio is not needed)
    writer.SetAudioOptions(False, "", 0, 0, openshot.LAYOUT_STEREO, 0)
    writer.Open()

    reader.info.fps.num =fps
    # Process and write all frames
    total_frames = int(reader.info.fps.num * reader.info.duration)
    print(f"Total Frames: {total_frames}")
    for frame_number in range(total_frames):
        try:
            frame = timeline.GetFrame(frame_number)
            if frame is not None:
                writer.WriteFrame(frame)
            else:
                print(f"⚠️ Skipping frame {frame_number} because it's None.")
        except Exception as e:
            print(f"❌ Error: {e}")
            response = {"error": str(e)}
            await websocket.send(json.dumps({'clientId': client_id, "response": response}))

        if frame_number % 500 == 0:
            print(f"✅ Rendered {frame_number} / {total_frames} frames...")

    writer.Close()  # Ensure writer is closed properly
    del writer  # Remove writer object from memory

    # await cleanup_client_resources(client_id)  # Cleanup after rendering
    response={"status": "Completed", "progress": 100,'reqdatajson':reqdata}
    await websocket.send(json.dumps({"clientId": client_id,"response": {"status": "Completed"},'response':response}))


async def process_data(websocket,client_id):
    """Simulate AI function sending progress updates."""
    print("Processing started...")
    total_steps = 100

    for i in range(1, total_steps + 1):
        progress = i
        time_left = (total_steps - i) * 0.5  # Simulated processing time
        response={"progress": progress, "time_left": time_left,'reqdatajson':{}}
        # Send progress update to the client
        message = json.dumps({"clientId": client_id,'response':response})
        await websocket.send(message)

        await asyncio.sleep(0.5)  # Simulate AI processing delay

    # Send completion message
    response={"progress": 100, "time_left": 0,'reqdatajson':{}}
    await websocket.send(json.dumps({"clientId": client_id,'response':response}))
    print("Processing completed.")


async def process_video(websocket,client_id,reqdata,processor):
    data = json.loads(reqdata)
    input_lang=data.get("src")
    clusters=data.get("clusters")
    video_path= os.path.join(UPLOAD_FOLDER, data.get("inputFile"))
    final_output_path= os.path.join(UPLOAD_FOLDER, data.get("outputFile"))
    try:
            print(f"process_video ")


            # Step 1: Extract Audio
            print(f"Extracting Audio",video_path)
            response={"status": "Extracting Audio", "progress": 10,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            audio_output_path = "output_audio.wav"
            AudioSegment.from_file(video_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")

            # Step 2: Process Audio & Extract Speech
            print(f"Processing Audio")
            response={"status": "Processing Audio", "progress": 15,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            predicted_segments, _ = await processor.extract_speech_segments_and_cluster(websocket,client_id,audio_output_path, "output_segments", n_clusters=int(clusters))

            # Step 3: Add Subtitles
            print(f"Adding Subtitles")
            response={"status": "Adding Subtitles", "progress": 60,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            await processor.add_subtitles_to_video(websocket,client_id, video_path, predicted_segments, final_output_path,reqdata)
            print(f"Completed Audio")

    except Exception as e:
            response={"status": "error", "message": str(e),'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))


async def process_video_translate(websocket,client_id,reqdata,processor):
    data = json.loads(reqdata)
    src=data.get("src")
    lang=data.get("lang") 
    clusters=data.get("clusters")
    input_path= os.path.join(UPLOAD_FOLDER, data.get("inputFile"))
    output_path= os.path.join(UPLOAD_FOLDER, data.get("outputFile"))
    try:
            print(f"process_video ")


            # Step 1: Extract Audio
            print(f"Extracting Audio")
            response={"status": "Extracting Audio", "progress": 10,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            audio_output_path = "output_audio.wav"
            AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")

            # Step 2: Process Audio & Extract Speech
            print(f"Processing Audio")
            response={"status": "Processing Audio", "progress": 15,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            predicted_segments, _ = await processor.extract_speech_segments_and_cluster(websocket,client_id, audio_output_path, "output_segments", n_clusters=int(clusters))

            # Step 3: Add Subtitles
            print(f"Adding Audio")
            response={"status": "Adding Subtitles", "progress": 60,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            await processor.add_subtitles_to_video_translate(websocket,client_id, input_path, predicted_segments, output_path,src,lang,reqdata)
            print(f"Completed Audio")

    except Exception as e:
            response={"status": "error", "message": str(e),'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))


   
async def handler(websocket):  # ✅ No 'path' argument
    """Handle new WebSocket connections."""
    model_path = "output/denoiser_model4.pth"
    processor = AudioVideoProcessor(model_path)
    try:
        # Wait for a message from the frontend
        async for message in websocket:
            datatext = json.loads(message)  # Parse incoming JSON
            client_id = datatext.get("clientId")  # Extract client ID
            datastr = datatext.get("message")
            print(f"data recevied {datastr}")
            data = json.loads(datastr)
            print(f"Received from Node.js (Client {client_id}): {datatext}")
            function_name = data.get("function")  # Get the function name from message
            reqdata = data.get("data")  # Get the function name from message
            if  function_name == "translate":
               await process_video_translate(websocket,client_id,reqdata,processor)
            elif function_name == "transcribe":
                await process_video(websocket,client_id,reqdata,processor)
            elif function_name == "applyBlur":
                await apply_blur_filter(websocket,client_id,reqdata)
            elif function_name == "applyBrightness":
                await apply_brightness_filter(websocket,client_id,reqdata)
            elif function_name == "applyContrast":
                await apply_contrast_filter(websocket, client_id, reqdata)
            elif function_name == "applyGrayscale":
                await apply_grayscale_filter(websocket, client_id, reqdata)
            elif function_name == "applySepia":
                await apply_sepia_filter(websocket, client_id, reqdata)
            elif function_name == "renderVideo":
                await render_video(websocket, client_id, reqdata)
            elif  function_name == "process_data":
                await process_data(websocket,client_id)
            else:
                # Handle unrecognized function
                response={"error": "Unknown function requested"}
                await websocket.send(json.dumps({'clientId':client_id,"response": response}))

    except websockets.exceptions.ConnectionClosed:
        print(f"⚠️ Client {client_id} disconnected.")
        await cleanup_client_resources(client_id)


async def cleanup_client_resources(client_id):
    """Safely closes and deletes all resources for a disconnected client."""
    if client_id in client_projects:
        project = client_projects.pop(client_id, None)
        if project:
            try:
                if "reader" in project and project["reader"]:
                    project["reader"].Close()  # Explicitly close reader
                    del project["reader"]

                # Delay timeline deletion to avoid premature cleanup
                await asyncio.sleep(0.5)  

                if "timeline" in project and project["timeline"]:
                    del project["timeline"]

                if "clip" in project:
                    del project["clip"]

                if "effects" in project:
                    del project["effects"]

                # Delay GC to ensure OpenShot objects are fully released
                await asyncio.sleep(1)
                gc.collect()

                print(f"✅ Safely cleaned up resources for Client {client_id}.")
            except Exception as e:
                print(f"⚠️ Error cleaning up resources for Client {client_id}: {e}")
async def main():
    print("Python WebSocket Server Started at ws://0.0.0.0:8765")
    
    # ✅ Wrap handler using functools.partial to remove 'path' argument
    server = websockets.serve(functools.partial(handler), "0.0.0.0", 8765)
    
    async with server:
        await asyncio.Future()  # Keep the server running

if __name__ == "__main__":
    asyncio.run(main())
