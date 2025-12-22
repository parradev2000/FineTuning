"""
Módulo para procesar diferentes tipos de documentos
"""
import json
import PyPDF2
from docx import Document
import os


def extract_text_from_pdf(file_path):
    """Extrae texto de un archivo PDF"""
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error al leer PDF {file_path}: {str(e)}")
    return text


def extract_text_from_docx(file_path):
    """Extrae texto de un archivo DOCX"""
    text = ""
    try:
        doc = Document(file_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        print(f"Error al leer DOCX {file_path}: {str(e)}")
    return text


def process_documents(file_paths):
    """
    Procesa los documentos subidos y los convierte en un formato adecuado para fine-tuning
    """
    training_data = []
    
    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Dividir el contenido en segmentos manejables
                segments = split_text(content)
                for segment in segments:
                    training_data.append({
                        'text': segment
                    })
        
        elif ext == '.pdf':
            content = extract_text_from_pdf(file_path)
            segments = split_text(content)
            for segment in segments:
                training_data.append({
                    'text': segment
                })
        
        elif ext == '.docx':
            content = extract_text_from_docx(file_path)
            segments = split_text(content)
            for segment in segments:
                training_data.append({
                    'text': segment
                })
        
        elif ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Asumiendo que el JSON contiene un array de textos
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, str):
                            training_data.append({'text': item})
                        elif isinstance(item, dict) and 'text' in item:
                            training_data.append(item)
                elif isinstance(data, dict) and 'text' in data:
                    training_data.append(data)
    
    return training_data


def split_text(text, max_length=1024):
    """
    Divide un texto largo en segmentos manejables
    """
    if not text:
        return []
    
    segments = []
    for i in range(0, len(text), max_length):
        segment = text[i:i+max_length]
        if segment.strip():  # Solo añadir segmentos que no estén vacíos
            segments.append(segment)
    return segments