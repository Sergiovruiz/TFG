import fitz

def extraer_texto_pdf(ruta_pdf):
    try:
        doc = fitz.open(ruta_pdf)
    except:
        raise ValueError(f"No se pudo abrir el PDF: {ruta_pdf}")

    texto = ""
    for pagina in doc:
        texto += pagina.get_text("text") + "\n"
    doc.close()

    if not texto.strip():
        raise ValueError("El PDF no contiene texto")

    return texto
