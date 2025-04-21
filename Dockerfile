# Use official Python image with necessary dependencies
FROM python:3.10
# Set environment variables for model storage
ENV TORCH_HOME="/models/torch"
ENV TRANSFORMERS_CACHE="/models/transformers"
# Set environment variable for model storage
ENV WHISPER_CACHE="/models/whisper"
# Set working directory
WORKDIR /app

# Copy the backend files
COPY backend/ /app/

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# # Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# # Download Whisper model at build time
RUN python -c "import whisper; whisper.load_model('medium')"
RUN pip install sentencepiece
# RUN pip install openshot-python

# Install system dependencies
RUN apt update && apt install -y \
    cmake build-essential python3 python3-pip \
    libsndfile1-dev libavformat-dev libavcodec-dev libavutil-dev \
    libswscale-dev libavdevice-dev libavfilter-dev libswresample-dev \
    libjsoncpp-dev libzmq3-dev qtbase5-dev frei0r-plugins frei0r-plugins-dev \
    libopencv-dev libqt5svg5-dev swig \
    libprotobuf-dev protobuf-compiler libqt5multimedia5  \
    libqt5multimedia5-plugins qtbase5-dev qtchooser \
    libasound2-dev  qt5-qmake qttools5-dev qttools5-dev-tools # ALSA sound library

# Fix missing FFmpeg libavresample (Debian 12 removed it, so use libswresample)
RUN apt install -y libavresample-dev || echo "Skipping libavresample-dev (not available on Debian 12+)"

# Fix missing babl library
RUN apt install -y babl-dev || echo "Skipping babl-dev (not available in some distros)"


# Install OpenShotAudio
RUN  cd libopenshot-audio && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && ldconfig

 # Install OpenShot libraries
RUN cd libopenshot && mkdir build && cd build && \
    cmake .. && make -j$(nproc) && make install && ldconfig 
# RUN   pip install sqlalchemy psycopg2-binary
# Download models at build time
# RUN python -c "import whisper; whisper.load_model('medium').save('/models/whisper-medium')"

# # Download and save all models (if not already saved)
# RUN python -c "from transformers import MarianMTModel; \
#     models = ['ar-en', 'ar-ru', 'en-ar', 'en-ru', 'ru-en', 'ru-ar']; \
#     base_model = 'Helsinki-NLP/opus-mt-'; \
#     [MarianMTModel.from_pretrained(base_model + lang).save_pretrained('models/mt-' + lang) for lang in models]"

# # Download and save all tokenizers
# RUN python -c "from transformers import MarianTokenizer; \
#     models = ['ar-en', 'ar-ru', 'en-ar', 'en-ru', 'ru-en', 'ru-ar']; \
#     base_model = 'Helsinki-NLP/opus-mt-'; \
#     [MarianTokenizer.from_pretrained(base_model + lang).save_pretrained('models/mt-' + lang) for lang in models]"
# Expose WebSocket and API ports
EXPOSE 8765

# Start Python backend
CMD ["python", "server.py"]
