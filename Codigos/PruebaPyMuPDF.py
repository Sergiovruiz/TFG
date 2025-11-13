import fitz  # PyMuPDF

doc = fitz.open("F:\TFG\PDFs\short-paper.pdf")
texto = ""
for page in doc:
    texto += page.get_text("text")
print(texto)