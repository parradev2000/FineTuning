# Fine-tuning Qwen2.5-7B-Instruct

Esta aplicación web permite hacer fine-tuning al modelo Qwen2.5-7B-Instruct utilizando tus propios datos almacenados en documentos locales.

## Requisitos

- Python 3.8 o superior
- CUDA (recomendado para entrenamiento eficiente)
- Al menos 16GB de RAM (recomendado 32GB)
- Espacio de almacenamiento suficiente para el modelo (~15GB)

## Instalación

1. Clona o crea este proyecto en tu entorno local
2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Si tienes el modelo Qwen2.5-7B-Instruct localmente, actualiza la ruta en `app.py`. Si no, el modelo se descargará automáticamente desde Hugging Face (requiere conexión a internet).

## Configuración del modelo local

Si tienes el modelo Qwen2.5-7B-Instruct almacenado localmente, sigue estos pasos:

1. Coloca la carpeta del modelo en el directorio `/workspace`
2. Actualiza la variable `model_name` en la función `fine_tune_model` en `app.py` con la ruta local al modelo:

```python
model_name = "./ruta/al/modelo/local"  # Cambia esto a la ruta real de tu modelo
```

## Uso

1. Ejecuta la aplicación:

```bash
python app.py
```

2. Abre tu navegador y visita `http://localhost:5000`
3. Sube tus documentos (formatos soportados: .txt, .json, .pdf, .docx)
4. La aplicación procesará tus documentos y comenzará el proceso de fine-tuning

## Formato de datos

La aplicación soporta los siguientes formatos de archivo:

- **.txt**: Texto plano
- **.json**: Archivos JSON con array de textos o diccionarios con campo "text"
- **.pdf**: Documentos PDF (limitado)
- **.docx**: Documentos de Word (limitado)

Para resultados óptimos, se recomienda usar archivos .txt o .json con el formato:

```json
[
  {
    "text": "Tu texto de entrenamiento aquí..."
  },
  {
    "text": "Otro ejemplo de texto..."
  }
]
```

## Características

- Interfaz web sencilla para subir documentos
- Uso de LoRA para fine-tuning eficiente en memoria
- Soporte para modelos grandes usando 8-bit
- Procesamiento de múltiples formatos de archivo

## Consideraciones

- El fine-tuning de modelos grandes requiere recursos significativos
- El proceso puede tomar varias horas dependiendo del tamaño de los datos
- Se recomienda usar una GPU con al menos 11GB de VRAM
- El uso de LoRA reduce significativamente los requisitos de memoria

## Personalización

Puedes ajustar los parámetros de entrenamiento en la función `fine_tune_model`:

- `num_train_epochs`: Número de épocas de entrenamiento
- `per_device_train_batch_size`: Tamaño del batch (ajusta según tu memoria disponible)
- `learning_rate`: Tasa de aprendizaje
- `r`, `lora_alpha`, `lora_dropout`: Parámetros de LoRA