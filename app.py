from flask import Flask, request, render_template, jsonify
import os
import json
import threading
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
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
                        updateStatus("Error de conexión: " + err, "error");
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
                            if (data.state !== "completed" && data.state !== "error") {
                                setTimeout(pollStatus, 2000);
                            }
                        }
                    })
                    .catch(function() { setTimeout(pollStatus, 3000); });
            }
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