from flask import Flask, request, render_template, jsonify
import os
import json
import threading
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from document_processor import process_documents

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Asegurarse de que exista la carpeta de uploads
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Estado global del entrenamiento
training_status = {
    'state': 'idle',  # idle, uploading, processing, training, completed, error
    'message': 'Esperando archivos...',
    'progress': 0
}

# Modelo cargado para inferencia
inference_model = None
inference_tokenizer = None
inference_model_name = None

# Detectar si ya existe un modelo fine-tuned de una sesión anterior
if os.path.exists('./fine_tuned_model/adapter_config.json'):
    training_status['state'] = 'completed'
    training_status['message'] = 'Modelo fine-tuned encontrado. Listo para chatear.'
    training_status['progress'] = 100

# Ruta principal
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fine-tuning Qwen2.5-7B-Instruct</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-form { margin-bottom: 30px; }
            .form-group { margin-bottom: 15px; }
            label { display: block; margin-bottom: 5px; }
            input[type="file"] { width: 100%; padding: 8px; }
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 4px; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            button:disabled { background-color: #6c757d; cursor: not-allowed; }
            .status { margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 4px; border: 1px solid #dee2e6; }
            .progress-bar-container { width: 100%; background-color: #e9ecef; border-radius: 4px; margin-top: 10px; height: 25px; display: none; }
            .progress-bar { height: 25px; background-color: #007bff; border-radius: 4px; text-align: center; line-height: 25px; color: white; font-size: 14px; transition: width 0.5s ease; }
            .status.training { border-color: #007bff; background-color: #e7f1ff; }
            .status.completed { border-color: #28a745; background-color: #d4edda; }
            .status.error { border-color: #dc3545; background-color: #f8d7da; }
            .chat-section { margin-top: 30px; display: none; border: 1px solid #dee2e6; border-radius: 4px; overflow: hidden; }
            .chat-header { background-color: #28a745; color: white; padding: 15px; }
            .chat-header h3 { margin: 0; }
            .chat-messages { height: 300px; overflow-y: auto; padding: 15px; background-color: #f8f9fa; }
            .chat-message { margin-bottom: 12px; padding: 10px 14px; border-radius: 8px; max-width: 85%; word-wrap: break-word; white-space: pre-wrap; }
            .chat-message.user { background-color: #007bff; color: white; margin-left: auto; }
            .chat-message.assistant { background-color: #e9ecef; color: #333; }
            .chat-message.system { background-color: #fff3cd; color: #856404; text-align: center; max-width: 100%; font-style: italic; }
            .chat-input-area { display: flex; padding: 15px; background-color: white; border-top: 1px solid #dee2e6; }
            .chat-input-area textarea { flex: 1; padding: 10px; border: 1px solid #ced4da; border-radius: 4px; resize: none; font-family: Arial, sans-serif; font-size: 14px; }
            .chat-input-area button { margin-left: 10px; white-space: nowrap; }
            .loading-dots::after { content: "."; animation: dots 1.5s steps(3, end) infinite; }
            @keyframes dots { 0% { content: "."; } 33% { content: ".."; } 66% { content: "..."; } }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fine-tuning Qwen2.5-7B-Instruct</h1>
            <p>Sube tus documentos para entrenar el modelo con tus datos.</p>
            
            <form class="upload-form" id="uploadForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="files">Selecciona tus documentos:</label>
                    <input type="file" id="files" name="files[]" multiple accept=".txt,.pdf,.docx,.json">
                </div>
                <button type="submit" id="submitBtn">Subir y Procesar</button>
            </form>
            
            <div class="status" id="status">
                <h3>Estado:</h3>
                <p id="statusMessage">Esperando archivos...</p>
                <div class="progress-bar-container" id="progressContainer">
                    <div class="progress-bar" id="progressBar">0%</div>
                </div>
            </div>
            
            <div class="chat-section" id="chatSection">
                <div class="chat-header">
                    <h3>Chat con el modelo entrenado</h3>
                </div>
                <div class="chat-messages" id="chatMessages">
                    <div class="chat-message system">El modelo ha sido entrenado con tus datos. Escribe un mensaje para interactuar con el.</div>
                </div>
                <div class="chat-input-area">
                    <textarea id="chatInput" rows="2" placeholder="Escribe tu mensaje aqui..."></textarea>
                    <button id="chatSendBtn" onclick="sendChat()">Enviar</button>
                </div>
            </div>
        </div>
        
        <script>
            document.getElementById("uploadForm").addEventListener("submit", function(e) {
                e.preventDefault();
                var formData = new FormData(this);
                var submitBtn = document.getElementById("submitBtn");
                submitBtn.disabled = true;
                submitBtn.textContent = "Procesando...";
                updateStatus("Subiendo archivos...", "training");
                
                fetch("/upload", { method: "POST", body: formData })
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        if (data.status === "started") {
                            updateStatus("Entrenamiento iniciado. Monitoreando progreso...", "training");
                            showProgress();
                            pollStatus();
                        } else if (data.status === "error") {
                            updateStatus("Error: " + data.message, "error");
                            submitBtn.disabled = false;
                            submitBtn.textContent = "Subir y Procesar";
                        }
                    })
                    .catch(function(err) {
                        updateStatus("Error de conexion: " + err, "error");
                        submitBtn.disabled = false;
                        submitBtn.textContent = "Subir y Procesar";
                    });
            });
            
            function updateStatus(message, statusClass) {
                var statusDiv = document.getElementById("status");
                document.getElementById("statusMessage").textContent = message;
                statusDiv.className = "status" + (statusClass ? " " + statusClass : "");
            }
            
            function showProgress() {
                document.getElementById("progressContainer").style.display = "block";
            }
            
            function pollStatus() {
                fetch("/status")
                    .then(function(response) { return response.json(); })
                    .then(function(data) {
                        var bar = document.getElementById("progressBar");
                        bar.style.width = data.progress + "%";
                        bar.textContent = data.progress + "%";
                        updateStatus(data.message, data.state === "completed" ? "completed" : data.state === "error" ? "error" : "training");
                        
                        if (data.state === "training" || data.state === "processing") {
                            setTimeout(pollStatus, 2000);
                        } else {
                            var submitBtn = document.getElementById("submitBtn");
                            submitBtn.disabled = false;
                            submitBtn.textContent = "Subir y Procesar";
                            if (data.state === "completed") {
                                showChat();
                            }
                            if (data.state !== "completed" && data.state !== "error") {
                                setTimeout(pollStatus, 2000);
                            }
                        }
                    })
                    .catch(function() { setTimeout(pollStatus, 3000); });
            }
            
            function showChat() {
                document.getElementById("chatSection").style.display = "block";
                document.getElementById("chatSection").scrollIntoView({ behavior: "smooth" });
            }
            
            function sendChat() {
                var input = document.getElementById("chatInput");
                var message = input.value.trim();
                if (!message) return;
                
                var sendBtn = document.getElementById("chatSendBtn");
                sendBtn.disabled = true;
                sendBtn.textContent = "Generando...";
                input.disabled = true;
                
                addChatMessage(message, "user");
                input.value = "";
                
                var loadingDiv = document.createElement("div");
                loadingDiv.className = "chat-message assistant loading-dots";
                loadingDiv.id = "loadingMsg";
                loadingDiv.textContent = "Generando respuesta";
                document.getElementById("chatMessages").appendChild(loadingDiv);
                scrollChat();
                
                fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ prompt: message })
                })
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    var loading = document.getElementById("loadingMsg");
                    if (loading) loading.remove();
                    
                    if (data.status === "success") {
                        addChatMessage(data.response, "assistant");
                    } else {
                        addChatMessage("Error: " + data.message, "system");
                    }
                })
                .catch(function(err) {
                    var loading = document.getElementById("loadingMsg");
                    if (loading) loading.remove();
                    addChatMessage("Error de conexion: " + err, "system");
                })
                .finally(function() {
                    sendBtn.disabled = false;
                    sendBtn.textContent = "Enviar";
                    input.disabled = false;
                    input.focus();
                });
            }
            
            function addChatMessage(text, role) {
                var div = document.createElement("div");
                div.className = "chat-message " + role;
                div.textContent = text;
                document.getElementById("chatMessages").appendChild(div);
                scrollChat();
            }
            
            function scrollChat() {
                var messages = document.getElementById("chatMessages");
                messages.scrollTop = messages.scrollHeight;
            }
            
            document.getElementById("chatInput").addEventListener("keydown", function(e) {
                if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendChat();
                }
            });
            
            // Check if model is already trained on page load
            fetch("/status")
                .then(function(response) { return response.json(); })
                .then(function(data) {
                    if (data.state === "completed") {
                        updateStatus(data.message, "completed");
                        showProgress();
                        var bar = document.getElementById("progressBar");
                        bar.style.width = "100%";
                        bar.textContent = "100%";
                        showChat();
                    }
                });
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_files():
    global training_status
    if 'files[]' not in request.files:
        return jsonify({'status': 'error', 'message': 'No se han seleccionado archivos'}), 400
    
    files = request.files.getlist('files[]')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'status': 'error', 'message': 'No se han seleccionado archivos válidos'}), 400
    
    training_status['state'] = 'uploading'
    training_status['message'] = 'Subiendo archivos...'
    training_status['progress'] = 5
    
    uploaded_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    training_status['state'] = 'processing'
    training_status['message'] = f'{len(uploaded_files)} archivo(s) subidos. Procesando documentos...'
    training_status['progress'] = 15
    
    # Procesar los archivos y preparar los datos para entrenamiento
    try:
        training_data = process_documents(uploaded_files)
        training_status['message'] = f'{len(training_data)} segmentos extraídos. Iniciando fine-tuning...'
        training_status['progress'] = 25
        
        # Iniciar el proceso de fine-tuning en un hilo separado
        thread = threading.Thread(target=run_fine_tuning, args=(training_data,))
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Fine-tuning iniciado con {len(training_data)} segmentos de entrenamiento',
            'training_samples': len(training_data)
        })
    except Exception as e:
        training_status['state'] = 'error'
        training_status['message'] = f'Error durante el proceso: {str(e)}'
        return jsonify({
            'status': 'error',
            'message': f'Error durante el proceso: {str(e)}'
        }), 500


@app.route('/status')
def get_status():
    return jsonify(training_status)


@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para chatear con el modelo fine-tuned"""
    global inference_model, inference_tokenizer, inference_model_name

    if training_status['state'] != 'completed':
        return jsonify({
            'status': 'error',
            'message': 'El modelo aun no ha sido entrenado. Sube documentos y entrena primero.'
        }), 400

    data = request.get_json()
    if not data or 'prompt' not in data:
        return jsonify({
            'status': 'error',
            'message': 'No se ha proporcionado un prompt.'
        }), 400

    prompt = data['prompt'].strip()
    if not prompt:
        return jsonify({
            'status': 'error',
            'message': 'El prompt esta vacio.'
        }), 400

    try:
        # Load the fine-tuned model if not already loaded
        if inference_model is None:
            inference_model, inference_tokenizer, inference_model_name = load_inference_model()

        response_text = generate_response(prompt)
        return jsonify({
            'status': 'success',
            'response': response_text,
            'model': inference_model_name
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error durante la generacion: {str(e)}'
        }), 500


def load_inference_model():
    """Carga el modelo fine-tuned para inferencia"""
    model_path = './fine_tuned_model'

    if not os.path.exists(model_path):
        raise FileNotFoundError('No se encontro el modelo fine-tuned en ' + model_path)

    if torch.cuda.is_available():
        base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        dtype = torch.float16
        device_map = "auto"
    else:
        base_model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        dtype = torch.float32
        device_map = None

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"torch_dtype": dtype}
    if device_map:
        load_kwargs["device_map"] = device_map

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **load_kwargs
    )

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer, base_model_name


def generate_response(prompt, max_new_tokens=256):
    """Genera una respuesta usando el modelo fine-tuned"""
    global inference_model, inference_tokenizer

    messages = [
        {"role": "system", "content": "Eres un asistente util entrenado con datos personalizados."},
        {"role": "user", "content": prompt}
    ]

    text = inference_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = inference_tokenizer(text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(inference_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = inference_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # Decode only the new tokens (skip the input)
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = inference_tokenizer.decode(new_tokens, skip_special_tokens=True)

    return response.strip()


def run_fine_tuning(training_data):
    """Ejecuta el fine-tuning en un hilo separado"""
    global training_status
    try:
        training_status['state'] = 'training'
        training_status['message'] = 'Cargando modelo y tokenizer...'
        training_status['progress'] = 30
        result = fine_tune_model(training_data)
        training_status['state'] = 'completed'
        training_status['message'] = f'Fine-tuning completado. {result["training_samples"]} muestras procesadas.'
        training_status['progress'] = 100
    except Exception as e:
        training_status['state'] = 'error'
        training_status['message'] = f'Error durante el entrenamiento: {str(e)}'



def fine_tune_model(training_data):
    """
    Realiza el fine-tuning del modelo Qwen2.5-7B-Instruct con los datos proporcionados
    """
    global training_status
    # Cargar el modelo y tokenizer
    # Usa Qwen2.5-0.5B-Instruct para entornos sin GPU, o Qwen2.5-7B-Instruct con GPU
    if torch.cuda.is_available():
        model_name = "Qwen/Qwen2.5-7B-Instruct"
        dtype = torch.float16
        device_map = "auto"
    else:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        dtype = torch.float32
        device_map = None
    
    training_status['message'] = f'Cargando tokenizer de {model_name}...'
    training_status['progress'] = 35
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Añadir token de padding si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_status['message'] = f'Cargando modelo {model_name}...'
    training_status['progress'] = 40
    
    load_kwargs = {
        "torch_dtype": dtype,
    }
    if device_map:
        load_kwargs["device_map"] = device_map
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **load_kwargs
    )
    
    # Preparar dataset
    texts = [item['text'] for item in training_data]
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized
    
    training_status['message'] = 'Tokenizando datos...'
    training_status['progress'] = 55
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    
    training_status['message'] = 'Configurando LoRA...'
    training_status['progress'] = 60
    
    # Configurar LoRA para fine-tuning eficiente
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    
    model = get_peft_model(model, peft_config)
    
    training_status['message'] = 'Iniciando entrenamiento...'
    training_status['progress'] = 70
    
    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=500,
        logging_steps=10,
        learning_rate=5e-5,
        logging_dir="./logs",
        report_to="none",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
    )
    
    training_status['message'] = 'Entrenando modelo...'
    training_status['progress'] = 75
    
    # Entrenar el modelo
    trainer.train()
    
    training_status['message'] = 'Guardando modelo fine-tuned...'
    training_status['progress'] = 95
    
    # Guardar el modelo fine-tuned
    trainer.save_model()
    
    return {
        'model_path': './fine_tuned_model',
        'training_samples': len(training_data),
        'status': 'completed'
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)