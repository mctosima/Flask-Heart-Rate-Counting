# Use a Python base image
FROM python:3.9.0

# Install system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libgl1 libxext6
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y opencv-python


# Set working directory
WORKDIR /app

# Copy the requirements.txt file first to leverage Docker cache
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app/

# Expose the port that Flask runs on (change this if your Flask app runs on a different port)
EXPOSE 80

# Start the Flask application
CMD ["gunicorn", "--worker-tmp-dir", "/dev/shm", "app:app"]
