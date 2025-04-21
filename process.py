# from classes.audioDenoiserCNN import AudioDenoiserCNN
from classes.index import AudioVideoProcessor
from pydub import AudioSegment
import sys
import ffmpeg
sys.stdout.reconfigure(encoding='utf-8')

def process_video(input_path,src, clusters, output_path):
    model_path = "output/denoiser_model4.pth"
    # ffmpeg.input(input_path).output(output_path, vf='eq=brightness=0.06:contrast=1.5:saturation=1.2').run()

    processor = AudioVideoProcessor(model_path)
    # input_path = "input3.mp4"
    audio_output_path = "output_audio.wav"
    # final_output_path = "final_video_with_audio.mp4"
    AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")
    y, sr = processor.read_audio(audio_output_path)
    predicted_segments, _ = processor.extract_speech_segments_and_cluster(audio_output_path, "output_segments", n_clusters=int(clusters),language=src)
    processor.add_subtitles_to_video(input_path, predicted_segments, output_path)
   

if __name__ == "__main__":
    import sys
    process_video(sys.argv[1], sys.argv[2],sys.argv[3],sys.argv[4])
