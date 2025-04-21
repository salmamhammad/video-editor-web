# from classes.audioDenoiserCNN import AudioDenoiserCNN
from classes.index import AudioVideoProcessor
from pydub import AudioSegment
import sys
import ffmpeg
sys.stdout.reconfigure(encoding='utf-8')

def denoise(input_path, output_path):
    model_path = "output/denoiser_model4.pth"
    # ffmpeg.input(input_path).output(output_path, vf='eq=brightness=0.06:contrast=1.5:saturation=1.2').run()

    processor = AudioVideoProcessor(model_path)
    # input_path = "input3.mp4"
    audio_output_path = "output_audio.wav"
    # final_output_path = "final_video_with_audio.mp4"
    # AudioSegment.from_file(input_path).set_frame_rate(16000).set_channels(1).export(audio_output_path, format="wav")
    noisy_waveform, sr = processor.read_audio(input_path)
    noisy_waveform =  processor.normalize_audio(noisy_waveform)

     # Denoise the audio
    denoised_waveform = processor.denoise_audio( noisy_waveform)
    denoised_waveform = processor.apply_low_pass_filter(denoised_waveform, sr)
    denoised_waveform = processor.adjust_gain(denoised_waveform)
    # Save the denoised audio
    processor.save_audio(output_path, denoised_waveform, sr)

if __name__ == "__main__":
    import sys
    denoise(sys.argv[1], sys.argv[2])
