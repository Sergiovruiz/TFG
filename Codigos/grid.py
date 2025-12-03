import numpy as np

def crear_matriz(modelos, documentos, temperaturas, prompts):
    return np.array(np.meshgrid(modelos, documentos, temperaturas, prompts)).T.reshape(-1, 4)
