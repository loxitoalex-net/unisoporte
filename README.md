# UniSoporte

Chatbot universitario construido con Streamlit, LangChain, LangGraph y Groq.

## Estructura

```text
UniSoporte_proyecto/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── run_windows.bat
├── run_linux_mac.sh
├── data/
│   └── tickets.json
├── .streamlit/
│   └── secrets.toml.example
└── docs/
    └── Documentacion_UniSoporte_estilo_Colab.ipynb
```

## Instalación local

1. Instala Python 3.10 o superior.
2. Abre una terminal dentro de esta carpeta.
3. Crea un entorno virtual:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

4. Instala las dependencias:

```bash
pip install -r requirements.txt
```

5. Copia `.env.example` como `.env` y coloca tu clave real de Groq:

```env
GROQ_API_KEY=tu_clave_real
GROQ_MODEL=llama-3.1-8b-instant
GROQ_BASE_URL=https://api.groq.com/openai/v1
```

6. Ejecuta la aplicación:

```bash
streamlit run app.py
```

También puedes usar `run_windows.bat` o `run_linux_mac.sh`.

## Dirección local

Streamlit normalmente abrirá:

```text
http://localhost:8501
```

## Archivos de datos

Los tickets creados por el agente se guardan en `data/tickets.json`. El archivo debe comenzar con una lista JSON vacía:

```json
[]
```

## Streamlit Community Cloud

En la configuración de Secrets puedes usar el contenido de `.streamlit/secrets.toml.example`. No subas claves reales al repositorio.
