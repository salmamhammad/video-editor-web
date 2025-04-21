import asyncio
import websockets
import json
import functools  # ✅ Import functools for partial function
from index2 import AudioVideoProcessor
from pydub import AudioSegment
import sys
import ffmpeg
import os
import time
import gc

import cv2
import numpy as np
import openshot 
os.environ["OMP_NUM_THREADS"] = "2"  # Limits OpenMP to 2 threads  
os.environ["OPENBLAS_NUM_THREADS"] = "2"  # Limits OpenBLAS (used by OpenShot)
sys.stdout.reconfigure(encoding='utf-8')
connected_clients = {}
UPLOAD_FOLDER = "/app/backend/"
client_projects = {}  # 🔹 Stores projects per client

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


async def process_video(websocket,client_id,reqdata):
    data = json.loads(reqdata)
    input_lang=data.get("src")
    clusters=data.get("clusters")
    video_path= os.path.join(UPLOAD_FOLDER, data.get("inputFile"))
    final_output_path= os.path.join(UPLOAD_FOLDER, data.get("outputFile"))
    try:
            print(f"process_video ")
            model_path = "output/denoiser_model4.pth"
            processor = AudioVideoProcessor(model_path)

            # Step 1: Extract Audio
            print(f"Extracting Audio",video_path)
            response={"status": "Extracting Audio", "progress": 10,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            audio_output_path = "output_audio.wav"
            AudioSegment.from_file(video_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")

            # Step 2: Process Audio & Extract Speech
            print(f"Processing Audio")
            response={"status": "Processing Audio", "progress": 30,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            predicted_segments, _ = await processor.extract_speech_segments_and_cluster(websocket,audio_output_path, "output_segments", n_clusters=int(clusters))

            # Step 3: Add Subtitles
            print(f"Adding Subtitles")
            response={"status": "Adding Subtitles", "progress": 70,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            await processor.add_subtitles_to_video(websocket,video_path, predicted_segments, final_output_path)
            print(f"Completed Audio")
            response={"status": "Completed", "progress": 100,'reqdatajson':reqdata}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
    except Exception as e:
            response={"status": "error", "message": str(e),'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))


async def process_video_translate(websocket,client_id,reqdata):
    data = json.loads(reqdata)
    src=data.get("src")
    lang=data.get("lang") 
    clusters=data.get("clusters")
    input_path= os.path.join(UPLOAD_FOLDER, data.get("inputFile"))
    output_path= os.path.join(UPLOAD_FOLDER, data.get("outputFile"))
    try:
            print(f"process_video ")
            model_path = "output/denoiser_model4.pth"
            processor = AudioVideoProcessor(model_path)

            # Step 1: Extract Audio
            print(f"Extracting Audio")
            response={"status": "Extracting Audio", "progress": 10,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            audio_output_path = "output_audio.wav"
            AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")

            # Step 2: Process Audio & Extract Speech
            print(f"Processing Audio")
            response={"status": "Processing Audio", "progress": 30,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            predicted_segments, _ = await processor.extract_speech_segments_and_cluster(websocket, audio_output_path, "output_segments", n_clusters=int(clusters))

            # Step 3: Add Subtitles
            print(f"Adding Audio")
            response={"status": "Adding Subtitles", "progress": 70,'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
            await processor.add_subtitles_to_video_translate(websocket, input_path, predicted_segments, output_path,src,lang)
            print(f"Completed Audio")
            response={"status": "Completed", "progress": 100,'reqdatajson':reqdata}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))
    except Exception as e:
            response={"status": "error", "message": str(e),'reqdatajson':{}}
            await websocket.send(json.dumps({"clientId": client_id,'response':response}))


   
async def handler(websocket):  # ✅ No 'path' argument
    """Handle new WebSocket connections."""
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
               await process_video_translate(websocket,client_id,reqdata)
            elif function_name == "transcribe":
                await process_video(websocket,client_id,reqdata)
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
