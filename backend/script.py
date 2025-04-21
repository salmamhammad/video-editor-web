import os

UPLOAD_FOLDER = "/app/backend/uploads"

# List all files in the shared volume
files = os.listdir(UPLOAD_FOLDER)

if files:
    print("Files in shared volume:", files)
else:
    print("No files found in shared volume")
