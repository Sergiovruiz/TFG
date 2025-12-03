from pymongo import MongoClient

def conectar_mongo():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["analisis_pdfs"]
    return db["resumenes"]

def guardar_resultado(coleccion, modelo, pdf, temperatura, prompt_id, prompt_texto, data):
    coleccion.insert_one({
        "archivo_pdf": pdf,
        "modelo": modelo,
        "temperatura": temperatura,
        "prompt_id": prompt_id,
        "prompt_texto": prompt_texto,
        "resultado": data
    })
