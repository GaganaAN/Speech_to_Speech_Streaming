from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
from moviepy import VideoFileClip
from gtts import gTTS
from pydub import AudioSegment
import subprocess
from dotenv import load_dotenv

app = Flask(__name__, static_folder='projects/static/uploads')

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


UPLOAD_FOLDER = 'projects/static/uploads'  
ALLOWED_EXTENSIONS = {'mp4', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Limit to 50 MB

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

google_chat_model = ChatGoogleGenerativeAI(
    model='gemini-1.5-flash',
    api_key=google_api_key
)

speech_to_text = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    device=-1
)

def transcribe_audio(file_path):
    result = speech_to_text(file_path)
    return result['text']

def translate_text(text, target_language):
    prompt = f"Translate the following text to {target_language}:\n\n{text}"
    response = google_chat_model.predict(prompt)
    return response

def convert_to_audio(text, output_file="output_audio.mp3"):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
    tts = gTTS(text, lang='hi')  # Change 'lang' as needed for the target language
    tts.save(output_path)
    return output_path

def adjust_audio_speed(audio_file, target_duration, output_file="adjusted_audio.mp3"):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_file)
    try:
        # Calculate the speed factor based on duration
        current_duration = AudioSegment.from_file(audio_file).duration_seconds
        speed_factor = current_duration / target_duration
        
        # Use ffmpeg to adjust the audio speed
        command = [
            "ffmpeg",
            "-i", audio_file,
            "-filter:a", f"atempo={speed_factor}",  # Adjust tempo (audio speed)
            "-vn",  # No video output
            output_path
        ]
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error in FFmpeg: {e}")
        return None


def convert_video_to_audio(video_file_path, output_audio_file="output_audio.mp3"):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_audio_file)
    video = VideoFileClip(video_file_path)
    video.audio.write_audiofile(output_path)
    return output_path
def replace_audio_with_ffmpeg(original_video_path, translated_audio_path, output_video_path="output_video.mp4"):
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_video_path)
    try:
        # Remove original audio (-an flag) and replace with the translated audio
        command = [
            "ffmpeg",
            "-i", original_video_path,        # Input video file
            "-i", translated_audio_path,      # Input translated audio file
            "-c:v", "copy",                   # Copy the video stream without re-encoding
            "-c:a", "aac",                    # Encode the new audio as AAC
            "-map", "0:v:0",                  # Map video stream from input 0 (original video)
            "-map", "1:a:0",                  # Map audio stream from input 1 (new audio) 
            "-shortest",                           # Disable the original audio
            output_path
        ]
        subprocess.run(command, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error in FFmpeg: {e}")
        return None


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        target_language = request.form['target_language']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            if filename.endswith('.mp4'):
                audio_file = convert_video_to_audio(file_path)
                transcription = transcribe_audio(audio_file)
                translated_text = translate_text(transcription, target_language)
                translated_audio_file = convert_to_audio(translated_text)
                video = VideoFileClip(file_path)
                adjusted_audio_file = adjust_audio_speed(translated_audio_file, video.audio.duration)
                final_video_file = replace_audio_with_ffmpeg(file_path, adjusted_audio_file)

                if final_video_file:
                    video_url = f'/static/uploads/{os.path.basename(final_video_file)}'
                    print(f"Final video URL: {video_url}")  # Debugging line
                    return render_template('index.html', video_url=video_url)
                else:
                    return "Error generating the final video."
            else:
                return "Unsupported file type."
    return render_template('index.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
