from flask import Flask, request, render_template, jsonify
import os
import json
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
            button { background-color: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            button:hover { background-color: #0056b3; }
            .status { margin-top: 20px; padding: 10px; background-color: #f8f9fa; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Fine-tuning Qwen2.5-7B-Instruct</h1>
            <p>Sube tus documentos para entrenar el modelo con tus datos.</p>
            
            <form class="upload-form" action="/upload" method="post" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="files">Selecciona tus documentos:</label>
                    <input type="file" id="files" name="files[]" multiple accept=".txt,.pdf,.docx,.json">
                </div>
                <button type="submit">Subir y Procesar</button>
            </form>
            
            <div class="status" id="status">
                <h3>Estado:</h3>
                <p>Esperando archivos...</p>
            </div>
        </div>
        
        <script>
            // Función para actualizar el estado
            function updateStatus(message) {
                document.getElementById('status').innerHTML = '<h3>Estado:</h3><p>' + message + '</p>';
            }
        </script>
    </body>
    </html>
    '''

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No se han seleccionado archivos'}), 400
    
    files = request.files.getlist('files[]')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No se han seleccionado archivos válidos'}), 400
    
    uploaded_files = []
    for file in files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filepath)
    
    # Procesar los archivos y preparar los datos para entrenamiento
    try:
        training_data = process_documents(uploaded_files)
        # Iniciar el proceso de fine-tuning
        result = fine_tune_model(training_data)
        return jsonify({
            'status': 'success',
            'message': 'Fine-tuning completado exitosamente',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error durante el proceso: {str(e)}'
        }), 500



def fine_tune_model(training_data):
    """
    Realiza el fine-tuning del modelo Qwen2.5-7B-Instruct con los datos proporcionados
    """
    # Cargar el modelo y tokenizer
    model_name = "Qwen/Qwen2.5-7B-Instruct"  # Puedes cambiarlo a la ruta local de tu modelo
    
    # Si tienes el modelo localmente, usa la ruta local en lugar del nombre de Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Añadir token de padding si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True  # Para manejar mejor la memoria
    )
    
    # Preparar dataset
    texts = [item['text'] for item in training_data]
    dataset = Dataset.from_dict({"text": texts})
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
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
    
    # Configuración de entrenamiento
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        save_steps=500,
        logging_steps=10,
        learning_rate=5e-5,
        logging_dir="./logs",
        report_to=None,  # Desactivar logging externo
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    
    # Entrenar el modelo
    trainer.train()
    
    # Guardar el modelo fine-tuned
    trainer.save_model()
    
    return {
        'model_path': './fine_tuned_model',
        'training_samples': len(training_data),
        'status': 'completed'
    }

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)