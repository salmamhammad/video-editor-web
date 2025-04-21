import ffmpeg

# Apply blur to the video
def apply_blur(input_video_path, output_video_path):
    ffmpeg.input(input_video_path).output(output_video_path, vf='boxblur=10:1').run()

def apply_color_filter(input_video_path, output_video_path):
    ffmpeg.input(input_video_path).output(output_video_path, vf='eq=brightness=0.06:contrast=1.5:saturation=1.2').run()


if __name__ == "__main__":
    import sys
    if(sys.argv[1]=="blur"):
       apply_blur(sys.argv[2], sys.argv[3])
    if(sys.argv[1]=="color"):
       apply_color_filter(sys.argv[2], sys.argv[3])