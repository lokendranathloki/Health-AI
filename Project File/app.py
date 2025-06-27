from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'HcqK0s0fvwEhsrwgE5JmcpnR8RcBnKh8'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load local Hugging Face model and tokenizer once at startup
tokenizer = AutoTokenizer.from_pretrained("./granite-3.3-2b-instruct")
model = AutoModelForCausalLM.from_pretrained("./granite-3.3-2b-instruct")

# --- Utility functions ---

def load_bmi_data():
    try:
        with open('health_bmi_analytics.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return []

def save_bmi_data(data):
    with open('health_bmi_analytics.json', 'w') as f:
        json.dump(data, f, indent=2)

def calculate_bmi_category(bmi):
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal weight"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obesity"

def get_chat_history_path():
    email = session.get('email', 'default')
    return os.path.join('chat_histories', f"{email}.json")

def load_chat_history():
    path = get_chat_history_path()
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return []

def save_chat_history(history):
    os.makedirs('chat_histories', exist_ok=True)
    path = get_chat_history_path()
    with open(path, 'w') as f:
        json.dump(history, f)

# --- Authentication ---

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        session['loggedin'] = True
        session['name'] = name
        session['email'] = email
        return redirect(url_for('welcome'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
def home():
    return redirect(url_for('login'))

# --- Welcome Page ---

@app.route('/welcome')
def welcome():
    if 'loggedin' in session:
        return render_template('welcome.html', name=session['name'])
    return redirect(url_for('login'))

# --- Chat with Local Model ---

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    chat_history = load_chat_history()
    response = None
    user_input = None
    user_name = session.get('name', 'User')

    greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening']

    if request.method == 'POST':
        user_input = request.form['user_input']
        chat_history.append({'role': 'user', 'content': user_input})

        # Check for greeting
        if user_input.strip().lower() in greetings:
            ai_reply = f"Hello {user_name}! How can I assist you with your health today?"
        else:
            prompt = (
                "You are Health AI, a helpful assistant that ONLY answers health, wellness, and medical questions. "
                "If the user greets you (like 'hi', 'hello'), greet them back and ask how you can help with their health. "
                "If the user asks about anything unrelated to health, politely refuse and remind them you only answer health-related questions.\n"
            )
            for msg in chat_history[-10:]:
                if msg['role'] == 'user':
                    prompt += f"{user_name}: {msg['content']}\n"
                else:
                    prompt += f"Health AI: {msg['content']}\n"
            prompt += "Health AI:"

            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=500)  # Increased for more detail
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            ai_reply = full_output.split("Health AI:")[-1].strip()

        chat_history.append({'role': 'ai', 'content': ai_reply})
        save_chat_history(chat_history)
        response = ai_reply

    return render_template('chat.html', response=response, chat_history=chat_history)

@app.route('/clear_chat')
def clear_chat():
    # Remove chat history file for this user
    path = get_chat_history_path()
    if os.path.exists(path):
        os.remove(path)
    return redirect(url_for('chat'))

# --- Image Upload (stub) ---

@app.route('/image', methods=['GET', 'POST'])
def image():
    result = None
    if request.method == 'POST':
        pass  # Add image upload and analysis logic here
    return render_template('image_upload.html', result=result)

# --- Disease Prediction ---

@app.route('/disease', methods=['GET', 'POST'])
def disease():
    with open('disease_predictions_large.json', 'r') as f:
        predictions = json.load(f)
    all_symptoms = sorted({symptom for entry in predictions for symptom in entry['symptoms']})
    result = None

    if request.method == 'POST':
        selected_symptoms = set(request.form.getlist('symptoms'))
        best_match = None
        best_match_count = 0
        confidence_order = {'High': 3, 'Medium': 2, 'Low': 1}

        for entry in predictions:
            entry_symptoms = set(entry['symptoms'])
            match_count = len(selected_symptoms & entry_symptoms)
            if match_count > best_match_count or (
                match_count == best_match_count and best_match and
                confidence_order.get(entry['confidence'], 0) > confidence_order.get(best_match['confidence'], 0)
            ):
                best_match = entry
                best_match_count = match_count

        if best_match and best_match_count > 0:
            result = {
                'disease': best_match['predicted_disease'],
                'confidence': best_match['confidence'],
                'symptoms': best_match['symptoms']
            }
        else:
            result = {'disease': 'No match found', 'confidence': '-', 'symptoms': list(selected_symptoms)}

    return render_template('disease.html', all_symptoms=all_symptoms, result=result)

# --- Treatment Plans ---

@app.route('/treatment', methods=['GET', 'POST'])
def treatment():
    with open('treatment_plans.json', 'r') as f:
        treatment_plans = json.load(f)
    diseases = [plan['disease'] for plan in treatment_plans]
    selected_treatment = None
    selected_disease = None

    if request.method == 'POST':
        selected_disease = request.form['disease']
        for plan in treatment_plans:
            if plan['disease'] == selected_disease:
                selected_treatment = plan['treatment']
                break

    return render_template(
        'treatment.html',
        diseases=diseases,
        selected_disease=selected_disease,
        selected_treatment=selected_treatment
    )

# --- BMI Analytics ---

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    bmi_data = load_bmi_data()
    new_entry = None
    ai_insight = None

    if request.method == 'POST':
        height_cm = float(request.form['height_cm'])
        weight_kg = float(request.form['weight_kg'])
        heart_rate = int(request.form['heart_rate'])
        blood_pressure = request.form['blood_pressure']
        blood_glucose = float(request.form['blood_glucose'])
        bmi = round(weight_kg / ((height_cm / 100) ** 2), 2)
        bmi_category = calculate_bmi_category(bmi)
        user_id = len(bmi_data) + 1
        entry = {
            "user_id": user_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "bmi": bmi,
            "bmi_category": bmi_category,
            "heart_rate": heart_rate,
            "blood_pressure": blood_pressure,
            "blood_glucose": blood_glucose
        }
        bmi_data.append(entry)
        save_bmi_data(bmi_data)
        new_entry = entry

        # AI-generated insights
        prompt = (
            f"User health data:\n"
            f"Height: {height_cm} cm\n"
            f"Weight: {weight_kg} kg\n"
            f"BMI: {bmi} ({bmi_category})\n"
            f"Heart Rate: {heart_rate} bpm\n"
            f"Blood Pressure: {blood_pressure}\n"
            f"Blood Glucose: {blood_glucose} mg/dL\n"
            f"Give a short summary of potential health concerns and recommendations."
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        ai_insight = tokenizer.decode(outputs[0], skip_special_tokens=True).split("\n")[0]

    # Prepare data for graphs
    user_ids = [entry.get("user_id", idx+1) for idx, entry in enumerate(bmi_data)]
    bmis = [entry.get("bmi", 0) for entry in bmi_data]
    categories = [entry.get("bmi_category", "Unknown") for entry in bmi_data]
    heart_rates = [entry.get("heart_rate", 0) for entry in bmi_data]
    blood_pressures = [entry.get("blood_pressure", "") for entry in bmi_data]
    blood_glucoses = [entry.get("blood_glucose", 0) for entry in bmi_data]
    dates = [entry.get("date", "") for entry in bmi_data]

    return render_template(
        'analytics.html',
        user_ids=user_ids,
        bmis=bmis,
        categories=categories,
        new_entry=new_entry,
        heart_rates=heart_rates,
        blood_pressures=blood_pressures,
        blood_glucoses=blood_glucoses,
        dates=dates,
        ai_insight=ai_insight
    )

if __name__ == '__main__':
    app.run(debug=True)
