import os

def leer_configuracion(ruta_config):
    # Estructuras donde se guardará la configuración
    modelos = []
    documentos = []
    temperaturas = []
    prompts = []

    seccion = None
    prompt_buffer = []

    # 1. Verificar archivo
    if not os.path.exists(ruta_config):
        raise FileNotFoundError(f"ERROR: No existe el archivo de configuración:\n{ruta_config}")

    with open(ruta_config, "r", encoding="utf-8") as f:
        for num_linea, linea in enumerate(f, start=1):
            linea = linea.strip()

            # Ignorar vacíos y comentarios
            if not linea or linea.startswith("#"):
                continue

            # Detectar secciones tipo [XXXX]
            if linea.startswith("[") and linea.endswith("]"):
                nombre_seccion = linea[1:-1].lower()

                # Aceptar PROMPT o PROMPTS
                if nombre_seccion in ("modelos", "documentos", "temperatura", "prompts"):
                    seccion = nombre_seccion
                else:
                    raise ValueError(f"ERROR:Sección desconocida {linea}")
                continue

            # Manejo de final de prompts
            if linea == "FIN_PROMPT":
                if not prompt_buffer:
                    raise ValueError(f"ERROR en línea {num_linea}: FIN_PROMPT sin contenido previo")
                prompts.append("\n".join(prompt_buffer))
                prompt_buffer = []
                continue

            # Acumulación de prompts
            if seccion == "prompts":
                prompt_buffer.append(linea)
                continue

            # Modelos
            if seccion == "modelos":
                modelos.append(linea.replace("\\", "/"))
                continue

            # Documentos
            if seccion == "documentos":
                documentos.append(linea.replace("\\", "/"))
                continue

            # Temperatura
            if seccion == "temperatura":
                try:
                    temperaturas.append(float(linea))
                except ValueError:
                    raise ValueError(f"ERROR: Temperatura inválida en línea {num_linea}: '{linea}'")
                continue

            # Si llega aquí, hay un error
            raise ValueError(f"ERROR: Línea fuera de sección válida → '{linea}'")

    # Validaciones finales
    if prompt_buffer:
        raise ValueError("ERROR: Un prompt no fue cerrado con FIN_PROMPT")

    if not modelos:
        raise ValueError("ERROR: No se definió ningún modelo en [MODELOS]")

    if not documentos:
        raise ValueError("ERROR: No se definió ningún documento en [DOCUMENTOS]")

    if not temperaturas:
        raise ValueError("ERROR: No se definió ninguna temperatura en [TEMPERATURA]")

    if not prompts:
        raise ValueError("ERROR: No se definió ningún prompt en [PROMPTS]")

    return modelos, documentos, temperaturas, prompts
