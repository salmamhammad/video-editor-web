from moviepy.editor import VideoFileClip

def process_video(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip = clip.resize(height=720)  # Example processing
    clip.write_videofile(output_path, codec="libx264")

if __name__ == "__main__":
    import sys
    process_video(sys.argv[1], sys.argv[2])
