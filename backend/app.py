import joblib
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from flask import request, send_file
from flask import Flask, render_template, Response, jsonify, redirect
import cv2
from gesture_engine import GestureProcessor
from collections import Counter
import pickle
import random
import os
import librosa
import joblib
from flask import request, jsonify
from pydub import AudioSegment
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from flask import session

app = Flask(__name__, template_folder='templates')
app.secret_key = "neurosense_ai_secret_123"

# Load handwriting model
handwriting_model = load_model("cnn_combined_model_best.h5")
print("Handwriting model input shape:", handwriting_model.input_shape)
# 🔥 FORCE FFMPEG PATH
os.environ["PATH"] += os.pathsep + r"C:\Users\pandu\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin"

AudioSegment.converter = r"C:\Users\pandu\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\Users\pandu\Downloads\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin\ffprobe.exe"

print("FFMPEG:", AudioSegment.converter)
print("FFPROBE:", AudioSegment.ffprobe)

smell_model = pickle.load(open("smell_pd_model.pkl","rb"))

#voice model
voice_model = joblib.load('parkinsons_model.sav')
scaler = joblib.load('scaler.sav')

# app = Flask(__name__, template_folder='templates')

results = {
    "voice": None,
    "smell": None,
    "handwriting": None
}

processor = GestureProcessor()

# 🔥 Serve movement page
@app.route('/movement')
def movement():
    return render_template('movement.html')

# 🔙 Back to dashboard (frontend)
@app.route('/back_to_main')
def back_to_main():
    return redirect("http://127.0.0.1:8000/dashboard.html")

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, _ = processor.process_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_gesture')
def get_gesture():
    if processor.buffer:
        stable_prediction = Counter(processor.buffer).most_common(1)[0][0]
        return jsonify({"gesture": stable_prediction})
    return jsonify({"gesture": "Searching..."})


smell_sequence = [
"Mango","Orange","Lemon","Coffee","Rose",
"Onion","Chocolate","Banana","Garlic","Mint",
"Apple","Soap","Perfume","Coconut","Ginger",
"Tea","Clove","Cinnamon","Vinegar","Turmeric"
]

smell_pool = smell_sequence + [
"Strawberry","Pineapple","Vanilla","Sandalwood"
]

def generate_questions():
    questions = []
    for smell in smell_sequence:
        options = random.sample(smell_pool, 3)
        if smell not in options:
            options.append(smell)
        random.shuffle(options)

        questions.append({
            "correct": smell,
            "options": options
        })
    return questions

# 🔥 Serve smell page

@app.route("/smell")
def smell():
    questions = generate_questions()
    return render_template("smell_index.html", questions=questions)

@app.route("/predict_smell", methods=["POST"])
def predict_smell():

    age = int(request.form["age"])
    gender = int(request.form["gender"])

    answers = []
    for i in range(20):
        ans = int(request.form.get(f"q{i}", 0))
        answers.append(ans)

    score = sum(answers)

    input_data = np.array([[age, gender, score]])

    prediction = smell_model.predict(input_data)
    prob = smell_model.predict_proba(input_data)

    risk = round(prob[0][1] * 100, 2)

    result = "High Parkinson Risk" if prediction[0] == 1 else "Low Parkinson Risk"
    results["smell"] = score / 20
    session["smell"] = results["smell"]

    # interpretation
    if score <= 5:
        smell_level = "Severe Smell Loss"
    elif score <= 8:
        smell_level = "Moderate Smell Loss"
    elif score <= 14:
        smell_level = "Mild Smell Loss"
    else:
        smell_level = "Normal Smell Function"

    if risk < 30:
        risk_level = "Low Risk"
    elif risk < 60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    return render_template(
        "smell_result.html",
        score=score,
        risk=risk,
        result=result,
        smell_level=smell_level,
        risk_level=risk_level
    )
  

@app.route('/download_report')
def download_report():
    return send_file("patient_report.pdf", as_attachment=True)

def create_graph(score):
    plt.figure()
    plt.bar(['Smell Score'], [score])
    plt.title("Smell Test Score")
    plt.savefig("static/score_graph.png")
    plt.close()
    
def generate_report(age, score, risk, result):
    file_name = "patient_report.pdf"
    styles = getSampleStyleSheet()

    story = []
    story.append(Paragraph("Parkinson Report", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Age: {age}", styles['Normal']))
    story.append(Paragraph(f"Score: {score}/20", styles['Normal']))
    story.append(Paragraph(f"Risk: {risk}%", styles['Normal']))
    story.append(Paragraph(result, styles['Normal']))

    pdf = SimpleDocTemplate(file_name)
    pdf.build(story)

    return file_name

#voice feature extraction
def extract_features(file_path):
    y, sr = librosa.load(file_path)

    pitches, _ = librosa.piptrack(y=y, sr=sr)
    avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 150.0
    rms = np.mean(librosa.feature.rms(y=y))

    features = [avg_pitch, avg_pitch+10, avg_pitch-10, 0.001, 0.0001, 0.002, 0.003, 0.006,
                rms, 0.2, 0.02, 0.03, 0.03, 0.06, 0.02, 20.0, 0.1, 0.5, -5.0, 0.2, 2.2, 0.25]

    return np.array(features).reshape(1, -1)

@app.route('/voice', methods=['GET'])
def voice_page():
    return render_template('voice.html')

@app.route('/predict_voice', methods=['POST'])
def predict_voice():

    file = request.files['audio']

    webm_path = "temp.webm"
    wav_path = "temp.wav"

    file.save(webm_path)
    print("File saved")

    # 🔥 Convert webm → wav
    audio = AudioSegment.from_file(webm_path, format="webm")
    audio.export(wav_path, format="wav")
    print("Converted to WAV")

    # 🔥 Load WAV (fast)
    y, sr = librosa.load(wav_path, sr=22050, duration=3)
    print("Audio loaded")

    features = extract_features(wav_path)
    print("Features extracted")

    scaled = scaler.transform(features)
    prediction = voice_model.predict(scaled)
    print("Prediction done")
    # ✅ STORE RESULT
    results["voice"] = 1 if prediction[0] == 1 else 0
    session["voice"] = results["voice"]  

    os.remove(webm_path)
    os.remove(wav_path)

    if prediction[0] == 0:
        result = "Healthy Voice ✅"
    else:
        result = "Parkinson Symptoms Detected 🚨"

    return jsonify({'result': result})

#Handwriting routes
@app.route('/handwriting')
def handwriting():
    return render_template('handwriting.html')

@app.route('/predict_handwriting', methods=['POST'])
def predict_handwriting():

    file = request.files['image']

    filepath = "temp_img.png"
    file.save(filepath)

    # 🔥 Correct preprocessing
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.reshape(img, (1, 128, 128, 1))

    prediction = handwriting_model.predict(img)
    results["handwriting"] = float(prediction[0][0])
    session["handwriting"] = results["handwriting"]
    os.remove(filepath)
 
    if prediction[0][0] > 0.5:
        result = "Parkinson Detected 🚨"
    else:
        result = "Healthy Handwriting ✅"

    return jsonify({"result": result})

@app.route('/final_result')
def final_result():

    # ✅ Load previous results if not tested again
    if results["voice"] is None:
        results["voice"] = session.get("voice")

    if results["smell"] is None:
        results["smell"] = session.get("smell")

    if results["handwriting"] is None:
        results["handwriting"] = session.get("handwriting")

    scores = []
    graph_values = []

    # 🎤 Voice
    if results["voice"] is not None:
        score = 0.8 if results["voice"] == 1 else 0.2
        scores.append(score)
        graph_values.append(score)
    else:
        graph_values.append(0)

    # 👃 Smell
    if results["smell"] is not None:
        smell = results["smell"]

        if smell < 0.3:
            score = 0.8
        elif smell < 0.6:
            score = 0.5
        else:
            score = 0.2

        scores.append(score)
        graph_values.append(score)
    else:
        graph_values.append(0)

    # ✍️ Handwriting
    if results["handwriting"] is not None:
        hw = results["handwriting"]

        if hw > 0.6:
            score = 0.8
        elif hw > 0.3:
            score = 0.5
        else:
            score = 0.2

        scores.append(score)
        graph_values.append(score)
    else:
        graph_values.append(0)

    # ❌ No test done
    if not scores:
        return "⚠️ Please complete at least one test"

    # ✅ Final calculation
    final_score = sum(scores) / len(scores)
    risk_percent = round(final_score * 100, 2)

    # ✅ Risk level
    if risk_percent < 30:
        level = "Low Risk"
    elif risk_percent < 60:
        level = "Moderate Risk"
    else:
        level = "High Risk"

    return render_template(
        "final_result.html",
        risk=risk_percent,
        level=level,
        graph_values=graph_values
    )
# confidence = float(prediction[0][0]) * 100

# if confidence > 50:
#     result = f"Parkinson Detected 🚨 ({confidence:.2f}%)"
# else:
#     result = f"Healthy Handwriting ✅ ({100-confidence:.2f}%)"


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)