# -*- coding: utf-8 -*-
"""UniSoporte: chatbot/agente web con Streamlit, LangChain, LangGraph y Groq."""
from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# -----------------------------------------------------------------------------
# Configuración visual
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="UniSoporte | Agente universitario",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 1.5rem;}
      .hero {
        padding: 1.25rem 1.4rem; border-radius: 18px;
        background: linear-gradient(135deg, rgba(40,90,180,.12), rgba(120,60,190,.08));
        border: 1px solid rgba(120,120,140,.22); margin-bottom: 1rem;
      }
      .hero h1 {margin: 0; font-size: 2rem;}
      .hero p {margin: .45rem 0 0 0; opacity: .8;}
      .small-note {font-size: .88rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

load_dotenv()


def get_secret(name: str, default: str | None = None) -> str | None:
    """Lee primero Streamlit Secrets y luego variables de entorno."""
    try:
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")
GROQ_BASE_URL = get_secret("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TICKETS_FILE = DATA_DIR / "tickets.json"
DATA_DIR.mkdir(exist_ok=True)
if not TICKETS_FILE.exists():
    TICKETS_FILE.write_text("[]", encoding="utf-8")

# -----------------------------------------------------------------------------
# Base de conocimiento simulada
# -----------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {
        "id": "KB-001",
        "title": "Restablecimiento de contraseña institucional",
        "keywords": ["contraseña", "clave", "password", "cuenta", "ingresar"],
        "content": (
            "Ingresar al portal de identidad, elegir 'Olvidé mi contraseña', "
            "escribir el código institucional y completar la verificación. "
            "Nunca se debe solicitar al usuario su contraseña ni su código MFA."
        ),
    },
    {
        "id": "KB-002",
        "title": "Problemas con el aula virtual",
        "keywords": ["aula virtual", "moodle", "curso", "no aparece", "acceso"],
        "content": (
            "Verificar el acceso al correo institucional, cerrar sesión, borrar "
            "cookies e ingresar nuevamente. Si un curso no aparece 24 horas "
            "después de la matrícula, registrar un ticket."
        ),
    },
    {
        "id": "KB-003",
        "title": "Incidente de ciberseguridad",
        "keywords": ["phishing", "hackeo", "virus", "cuenta comprometida", "correo sospechoso"],
        "content": (
            "No abrir enlaces ni archivos sospechosos. Cambiar la contraseña "
            "desde un dispositivo seguro, cerrar sesiones y escalar el caso "
            "al equipo de seguridad."
        ),
    },
    {
        "id": "KB-004",
        "title": "Falla de equipos en aulas",
        "keywords": ["proyector", "computadora", "audio", "pantalla", "aula"],
        "content": (
            "Solicitar edificio, aula y horario. Verificar energía y fuente de "
            "entrada. Si la clase está en curso o próxima, crear un ticket de "
            "prioridad alta."
        ),
    },
    {
        "id": "KB-005",
        "title": "Problemas de matrícula",
        "keywords": ["matrícula", "vacante", "horario", "prerrequisito", "pago"],
        "content": (
            "Verificar periodo de matrícula, deudas, vacantes, cruces de horario "
            "y prerrequisitos. Los casos administrativos requieren ticket."
        ),
    },
]

SERVICE_STATUS = {
    "aula virtual": "operativo",
    "correo institucional": "operativo",
    "matrícula": "operativo",
    "portal de identidad": "operativo",
    "biblioteca digital": "mantenimiento programado",
}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def search_documents(query: str) -> list[dict]:
    words = set(normalize_text(query).split())
    ranked: list[tuple[int, dict]] = []
    for doc in KNOWLEDGE_BASE:
        searchable = normalize_text(
            " ".join([doc["title"], " ".join(doc["keywords"]), doc["content"]])
        )
        score = sum(1 for word in words if word in searchable)
        if score:
            ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [doc for _, doc in ranked[:3]]

# -----------------------------------------------------------------------------
# Herramientas del agente
# -----------------------------------------------------------------------------
class TicketInput(BaseModel):
    title: str = Field(description="Título corto del incidente")
    description: str = Field(description="Descripción completa")
    category: str = Field(description="Categoría del incidente")
    priority: Literal["baja", "media", "alta", "critica"]
    user_email: str = Field(default="no_proporcionado")
    location: str = Field(default="no_proporcionada")


@tool
def search_knowledge_base(query: str) -> str:
    """Busca procedimientos oficiales de soporte universitario."""
    return json.dumps(search_documents(query), ensure_ascii=False)


@tool
def check_service_status(service: str) -> str:
    """Consulta el estado simulado de un servicio institucional."""
    normalized = normalize_text(service)
    for service_name, status in SERVICE_STATUS.items():
        if service_name in normalized or normalized in service_name:
            return json.dumps(
                {"service": service_name, "status": status}, ensure_ascii=False
            )
    return json.dumps(
        {"service": service, "status": "servicio no reconocido"},
        ensure_ascii=False,
    )


@tool(args_schema=TicketInput)
def create_support_ticket(
    title: str,
    description: str,
    category: str,
    priority: str,
    user_email: str = "no_proporcionado",
    location: str = "no_proporcionada",
) -> str:
    """Crea un ticket cuando el caso requiere intervención humana."""
    try:
        tickets = json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        tickets = []

    ticket = {
        "ticket_id": f"INC-{uuid.uuid4().hex[:8].upper()}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "abierto",
        "title": title,
        "description": description,
        "category": category,
        "priority": priority,
        "user_email": user_email,
        "location": location,
    }
    tickets.append(ticket)
    TICKETS_FILE.write_text(
        json.dumps(tickets, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return json.dumps(ticket, ensure_ascii=False)


@tool
def request_human_escalation(reason: str, priority: str) -> str:
    """Escala un incidente grave a un especialista humano."""
    return json.dumps(
        {
            "escalation_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
            "priority": priority,
            "reason": reason,
            "status": "pendiente de revisión humana",
        },
        ensure_ascii=False,
    )


TOOLS = [
    search_knowledge_base,
    check_service_status,
    create_support_ticket,
    request_human_escalation,
]

SYSTEM_PROMPT = (
    "Eres UniSoporte, agente de mesa de ayuda universitaria. "
    "Responde siempre en español claro, profesional y breve. "
    "Antes de indicar procedimientos institucionales usa search_knowledge_base. "
    "Usa check_service_status cuando pregunten si un servicio funciona. "
    "Crea un ticket solo cuando se requiera intervención humana y haya datos "
    "suficientes. Nunca solicites contraseñas ni códigos MFA. Los incidentes "
    "de ciberseguridad deben escalarse. No inventes resultados de herramientas. "
    "Nunca muestres etiquetas <function=...>, JSON de llamadas internas ni nombres de "
    "herramientas al usuario; ejecuta la herramienta y después responde normalmente."
)

@st.cache_resource(show_spinner=False)
def build_agent(api_key: str, model: str, base_url: str):
    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
        streaming=True,
    )
    return create_react_agent(llm, tools=TOOLS, prompt=SystemMessage(content=SYSTEM_PROMPT))

TOOL_MARKUP_RE = re.compile(
    r"<function=(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)>(?P<args>\{.*?\})</function>",
    re.DOTALL,
)


def extract_answer(result: dict) -> str:
    """Obtiene únicamente la respuesta final visible del agente."""
    messages = result.get("messages", [])
    for message in reversed(messages):
        if not isinstance(message, AIMessage) or not message.content:
            continue
        # Ignorar mensajes intermedios que todavía contienen llamadas estructuradas.
        if getattr(message, "tool_calls", None):
            continue
        content = message.content if isinstance(message.content, str) else str(message.content)
        if content.strip():
            return content
    return "No pude generar una respuesta. Intenta reformular la consulta."


def repair_leaked_tool_markup(answer: str, llm: ChatOpenAI) -> str:
    """Ejecuta una llamada textual accidental y genera una respuesta legible."""
    match = TOOL_MARKUP_RE.search(answer)
    if not match:
        return answer

    tool_map = {tool.name: tool for tool in TOOLS}
    tool_name = match.group("name")
    selected_tool = tool_map.get(tool_name)
    if selected_tool is None:
        return "No pude procesar correctamente esa consulta. Intenta formularla nuevamente."

    try:
        arguments = json.loads(match.group("args"))
        tool_result = selected_tool.invoke(arguments)
        final = llm.invoke([
            SystemMessage(content=(
                "Convierte el resultado de la herramienta en una respuesta final breve, clara "
                "y en español. No menciones herramientas, funciones, JSON ni procesos internos."
            )),
            HumanMessage(content=(
                f"Consulta del usuario: {st.session_state.messages[-1]['content']}\n\n"
                f"Resultado disponible: {tool_result}"
            )),
        ])
        return final.content if isinstance(final.content, str) else str(final.content)
    except Exception:
        return "No pude procesar correctamente esa consulta. Intenta formularla nuevamente."

def to_langchain_messages(history: list[dict]) -> list:
    messages = []
    for item in history:
        if item["role"] == "user":
            messages.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            messages.append(AIMessage(content=item["content"]))
    return messages

# -----------------------------------------------------------------------------
# Interfaz
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <h1>🎓 UniSoporte</h1>
      <p>Agente inteligente para consultas universitarias, incidencias y tickets.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuración")
    st.caption(f"Modelo: `{GROQ_MODEL}`")
    st.markdown(
        "**Ejemplos de consulta**\n\n"
        "- Olvidé mi contraseña institucional.\n"
        "- El proyector del aula no enciende.\n"
        "- ¿Está funcionando el aula virtual?\n"
        "- Recibí un correo sospechoso."
    )
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown(
        '<div class="small-note">No compartas contraseñas, códigos MFA ni datos bancarios.</div>',
        unsafe_allow_html=True,
    )

if not GROQ_API_KEY:
    st.error("Falta configurar la clave `GROQ_API_KEY`.")
    st.info(
        "En tu computadora crea un archivo `.env`. En Streamlit Cloud entra a "
        "**App → Settings → Secrets** y agrega:\n\n"
        '`GROQ_API_KEY = "tu_clave"`'
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "¡Hola! Soy **UniSoporte**. Describe tu consulta o incidente "
                "universitario y te ayudaré con los pasos correspondientes."
            ),
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Escribe tu consulta de soporte...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_box = st.empty()
        with st.spinner("Analizando la solicitud..."):
            try:
                agent = build_agent(GROQ_API_KEY, GROQ_MODEL, GROQ_BASE_URL)
                history = to_langchain_messages(st.session_state.messages[-12:])
                result = agent.invoke({"messages": history})
                answer = extract_answer(result)
                # Protección para modelos que emiten una llamada como texto XML.
                if TOOL_MARKUP_RE.search(answer):
                    fallback_llm = ChatOpenAI(
                        model=GROQ_MODEL,
                        api_key=GROQ_API_KEY,
                        base_url=GROQ_BASE_URL,
                        temperature=0,
                    )
                    answer = repair_leaked_tool_markup(answer, fallback_llm)
            except Exception as exc:
                answer = (
                    "No pude completar la consulta. Verifica la clave de Groq, "
                    "la conexión a internet y las dependencias del proyecto.\n\n"
                    f"**Detalle técnico:** `{type(exc).__name__}: {exc}`"
                )
        response_box.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
