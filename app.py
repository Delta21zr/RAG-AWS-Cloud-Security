import os
import time
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import (
    MAX_HISTORY_MESSAGES,
    MAX_QUESTION_LENGTH,
    configurar_cadena_rag,
    procesar_y_agregar_pdf,
    sanitizar_nombre_pdf,
)

# 1. Diccionario de internacionalización (Español / English)
I18N = {
    "es": {
        "page_title": "AWS Security — Inteligencia Documental",
        "top_brand": "AWS Security Intelligence · RAG Local",
        "sidebar_brand": "AWS Cloud Security",
        "sidebar_title": "Documentation RAG",
        "new_session": "Nueva sesión",
        "llm_engine": "Motor LLM",
        "llm_tooltip_title": "Large Language Model:",
        "llm_tooltip_desc": "Modelo de lenguaje ejecutado localmente mediante Ollama. Redacta las respuestas basándose en los fragmentos recuperados sin enviar datos a internet.",
        "llm_value": "Llama 3.1 (Local)",
        "vector_index": "Índice vectorial",
        "vector_tooltip_title": "Base de Datos Vectorial (ChromaDB):",
        "vector_tooltip_desc": "Total de fragmentos de texto extraídos de los PDFs y convertidos en vectores matemáticos multilingües para búsqueda semántica.",
        "vectors_label": "vectores",
        "retrieval": "Recuperación",
        "retrieval_tooltip_title": "Búsqueda Híbrida + Cross-Encoder:",
        "retrieval_tooltip_desc": "Combina similitud vectorial (ChromaDB) con palabras clave exactas (BM25) mediante Reciprocal Rank Fusion y re-ordena los candidatos con un modelo Cross-Encoder multilingüe.",
        "retrieval_value": "Híbrida + Rerank",
        "active_memory": "Memoria activa",
        "memory_tooltip_title": "Ventana Deslizante (Sliding Window):",
        "memory_tooltip_desc": f"Cantidad de mensajes recientes que el sistema recuerda para entender preguntas de seguimiento, limitada a {MAX_HISTORY_MESSAGES} para proteger la memoria RAM.",
        "messages_label": "msjs",
        "temperature": "Temperatura",
        "temp_tooltip_title": "Control de Creatividad:",
        "temp_tooltip_desc": "El valor 0.0 elimina la aleatoriedad del modelo, obligándolo a dar respuestas técnicas precisas, deterministas y sin alucinaciones.",
        "temp_value": "0.0 (Determinista)",
        "ingest_expander": "Ingesta de documentos (PDF)",
        "ingest_uploader": "Seleccione uno o más archivos PDF",
        "ingest_btn": "Indexar en base de datos",
        "ingest_processing": "Procesando e indexando PDF(s)...",
        "ingest_success": "Se indexaron {chunks} fragmentos de {docs} documento(s).",
        "ingest_invalid": "Archivo inválido ({name}): no es un PDF válido.",
        "ingest_error": "Error al procesar {name}: {err}",
        "docs_in_repo": "Documentos en repositorio ({count}):",
        "glossary_expander": "Glosario de términos",
        "empty_alert_title": "Índice vectorial vacío.",
        "empty_alert_desc": "Ejecute <code>python ingestar_datos.py</code> en la terminal para procesar los documentos de la carpeta <code>data/</code>.",
        "empty_kicker": "Sistema de Consulta Privado",
        "empty_heading": "Asistente Técnico de Seguridad en AWS",
        "empty_sub": "Realice consultas específicas sobre control de acceso (IAM), seguridad de red (VPC, Security Groups, NACLs), cifrado (KMS, S3), monitoreo (CloudTrail, CloudWatch, GuardDuty) y respuesta a incidentes. Las respuestas se generan exclusivamente a partir de la documentación local verificada.",
        "role_user": "Consulta",
        "role_assistant": "Respuesta",
        "chat_input_placeholder": "Escriba su consulta técnica sobre seguridad en AWS...",
        "analyzing_spinner": "Analizando documentación...",
        "contextualized_query": "Consulta contextualizada:",
        "doc_references": "Referencias documentales",
        "page_label": "Página",
        "connection_error": "Error de conexión con el motor de inferencia local (Ollama). Verifique que el servicio esté activo y ejecutando el modelo 'llama3.1'.",
        "glossary_items": [
            ("RAG (Retrieval-Augmented Generation)", "Arquitectura que busca primero en tus documentos PDF antes de generar una respuesta con IA."),
            ("Embeddings Multilingües", "Representación numérica del significado de un texto que permite buscar indistintamente en español o inglés dentro de PDFs."),
            ("IAM (Identity and Access Management)", "Servicio de AWS para controlar usuarios, roles, políticas y el principio de privilegio mínimo."),
            ("Security Group vs. NACL", "Firewalls virtuales en AWS: Security Group opera a nivel de instancia EC2 (con estado / stateful) y NACL a nivel de subred (sin estado / stateless)."),
            ("CloudTrail / CloudWatch / GuardDuty", "Tríada de monitoreo: CloudTrail audita llamadas a la API, CloudWatch supervisa métricas/logs operativos y GuardDuty detecta amenazas con IA."),
            ("CSE vs. SSE (Cifrado)", "Client-Side Encryption cifra los datos antes de subirlos a AWS; Server-Side Encryption los cifra automáticamente en los servidores de AWS.")
        ]
    },
    "en": {
        "page_title": "AWS Security — Documentation Intelligence",
        "top_brand": "AWS Security Intelligence · Local RAG",
        "sidebar_brand": "AWS Cloud Security",
        "sidebar_title": "Documentation RAG",
        "new_session": "New session",
        "llm_engine": "LLM Engine",
        "llm_tooltip_title": "Large Language Model:",
        "llm_tooltip_desc": "Locally executed language model via Ollama. Crafts answers based strictly on retrieved chunks without sending data to the internet.",
        "llm_value": "Llama 3.1 (Local)",
        "vector_index": "Vector Index",
        "vector_tooltip_title": "Vector Database (ChromaDB):",
        "vector_tooltip_desc": "Total text chunks extracted from PDFs and converted into multilingual mathematical vectors for semantic retrieval.",
        "vectors_label": "vectors",
        "retrieval": "Retrieval",
        "retrieval_tooltip_title": "Hybrid Search + Cross-Encoder:",
        "retrieval_tooltip_desc": "Combines vector similarity (ChromaDB) with exact keywords (BM25) via Reciprocal Rank Fusion, then re-ranks candidates with a multilingual Cross-Encoder.",
        "retrieval_value": "Hybrid + Rerank",
        "active_memory": "Active Memory",
        "memory_tooltip_title": "Sliding Window:",
        "memory_tooltip_desc": f"Number of recent conversation messages remembered for follow-up questions, bounded to {MAX_HISTORY_MESSAGES} to protect RAM.",
        "messages_label": "msgs",
        "temperature": "Temperature",
        "temp_tooltip_title": "Creativity Control:",
        "temp_tooltip_desc": "A 0.0 value eliminates randomness, ensuring deterministic, accurate technical answers without hallucination.",
        "temp_value": "0.0 (Deterministic)",
        "ingest_expander": "Document Ingestion (PDF)",
        "ingest_uploader": "Select one or more PDF files",
        "ingest_btn": "Index into database",
        "ingest_processing": "Processing and indexing PDF(s)...",
        "ingest_success": "Indexed {chunks} chunks from {docs} document(s).",
        "ingest_invalid": "Invalid file ({name}): not a valid PDF file.",
        "ingest_error": "Error processing {name}: {err}",
        "docs_in_repo": "Documents in repository ({count}):",
        "glossary_expander": "Terminology Glossary",
        "empty_alert_title": "Vector index is empty.",
        "empty_alert_desc": "Run <code>python ingestar_datos.py</code> in your terminal to process documents in the <code>data/</code> folder.",
        "empty_kicker": "Private Retrieval System",
        "empty_heading": "AWS Security Technical Assistant",
        "empty_sub": "Ask technical questions about access control (IAM), network security (VPC, Security Groups, NACLs), encryption (KMS, S3), monitoring (CloudTrail, CloudWatch, GuardDuty), and incident response. Answers are generated exclusively from verified local documentation.",
        "role_user": "Query",
        "role_assistant": "Response",
        "chat_input_placeholder": "Ask a technical question about AWS security...",
        "analyzing_spinner": "Analyzing documentation...",
        "contextualized_query": "Contextualized query:",
        "doc_references": "Documentary references",
        "page_label": "Page",
        "connection_error": "Connection error with the local inference engine (Ollama). Verify that the service is running with the 'llama3.1' model.",
        "glossary_items": [
            ("RAG (Retrieval-Augmented Generation)", "Architecture that retrieves relevant PDF chunks before generating answers with an AI model."),
            ("Multilingual Embeddings", "Numerical representation of text meaning enabling search in Spanish or English across PDF documentation."),
            ("IAM (Identity and Access Management)", "AWS service to manage identities, roles, policies, and enforce the principle of least privilege."),
            ("Security Group vs. NACL", "AWS virtual firewalls: Security Groups operate at EC2 instance level (stateful), while NACLs operate at subnet level (stateless)."),
            ("CloudTrail / CloudWatch / GuardDuty", "Monitoring triad: CloudTrail audits API calls, CloudWatch tracks operational metrics/logs, and GuardDuty detects AI-driven threats."),
            ("CSE vs. SSE (Encryption)", "Client-Side Encryption encrypts data prior to uploading to AWS; Server-Side Encryption automatically encrypts objects on AWS storage servers.")
        ]
    }
}

# Inicializar idioma por defecto
if "idioma" not in st.session_state:
    st.session_state.idioma = "es"

t = I18N[st.session_state.idioma]

# 2. Configuración de la página (Sin emojis, diseño sobrio y ejecutivo)
st.set_page_config(
    page_title=t["page_title"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. CSS minimalista con tooltips interactivos estilo (i) y glosario técnico
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Ocultar elementos nativos de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Contenedor central de lectura cómoda */
    .block-container {
        max-width: 820px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 6rem !important;
    }

    /* Barra superior con selector de idioma */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1.25rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid #1e222a;
    }

    .top-brand {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6e7681;
    }

    /* Barra lateral sobria */
    [data-testid="stSidebar"] {
        background-color: #0f1115;
        border-right: 1px solid #1e222a;
    }

    .sidebar-brand {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #8b949e;
        margin-bottom: 0.25rem;
    }

    .sidebar-title {
        font-size: 1.15rem;
        font-weight: 500;
        color: #f0f6fc;
        margin-bottom: 1.5rem;
    }

    .meta-panel {
        border-top: 1px solid #1e222a;
        padding-top: 1.1rem;
        margin-top: 1.25rem;
        margin-bottom: 1.25rem;
    }

    .meta-row {
        position: relative;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 0.8rem;
        padding: 0.5rem 0;
        border-bottom: 1px solid #161920;
    }

    .meta-label-group {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        color: #7d8590;
    }

    .info-trigger {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        color: #6e7681;
        cursor: help;
        transition: color 0.15s ease;
    }

    .info-trigger:hover {
        color: #c9d1d9;
    }

    /* Tooltip interactivo con CSS puro */
    .info-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        left: 0;
        bottom: 130%;
        width: 250px;
        background-color: #1c2128;
        color: #c9d1d9;
        font-size: 0.74rem;
        line-height: 1.45;
        padding: 0.65rem 0.8rem;
        border-radius: 6px;
        border: 1px solid #30363d;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
        z-index: 1000;
        transition: opacity 0.18s ease, visibility 0.18s ease;
        pointer-events: none;
        white-space: normal;
    }

    .info-tooltip strong {
        color: #f0f6fc;
        display: block;
        margin-bottom: 0.25rem;
        font-weight: 600;
    }

    .meta-row:hover .info-tooltip {
        visibility: visible;
        opacity: 1;
    }

    .meta-value {
        font-family: 'JetBrains Mono', monospace;
        color: #c9d1d9;
        font-size: 0.78rem;
    }

    /* Glosario en acordeón lateral */
    .glossary-item {
        padding: 0.55rem 0;
        border-bottom: 1px solid #1e222a;
    }

    .glossary-term {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.74rem;
        font-weight: 500;
        color: #e6edf3;
        margin-bottom: 0.2rem;
    }

    .glossary-desc {
        font-size: 0.75rem;
        color: #8b949e;
        line-height: 1.45;
    }

    /* Estado inicial vacío */
    .empty-state {
        text-align: left;
        margin-top: 10vh;
        margin-bottom: 4vh;
        padding: 2.5rem;
        border: 1px solid #1e222a;
        border-radius: 8px;
        background-color: #11141a;
    }

    .empty-kicker {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #6e7681;
        margin-bottom: 0.75rem;
    }

    .empty-heading {
        font-size: 1.75rem;
        font-weight: 400;
        color: #f0f6fc;
        letter-spacing: -0.02em;
        margin-bottom: 0.75rem;
    }

    .empty-sub {
        font-size: 0.92rem;
        color: #8b949e;
        line-height: 1.6;
    }

    /* Ocultar avatares por defecto de Streamlit para lograr un estilo tipográfico limpio */
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    .stChatMessage .st-emotion-cache-1p1m4ay {
        display: none !important;
    }

    /* Mensajes del chat */
    [data-testid="stChatMessage"] {
        background-color: transparent !important;
        border-bottom: 1px solid #181b22;
        padding: 1.5rem 0.5rem !important;
        border-radius: 0 !important;
    }

    .role-label {
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6e7681;
        margin-bottom: 0.5rem;
    }

    /* Tarjetas de referencias documentales */
    .ref-rephrased {
        font-size: 0.82rem;
        color: #9ca3af;
        background-color: #131720;
        border-left: 2px solid #4b5563;
        padding: 0.6rem 0.9rem;
        margin-bottom: 1rem;
    }

    .ref-card {
        border: 1px solid #1e222a;
        background-color: #0d0f14;
        border-radius: 6px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.65rem;
    }

    .ref-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.73rem;
        color: #9ca3af;
        margin-bottom: 0.45rem;
        display: flex;
        justify-content: space-between;
    }

    .ref-excerpt {
        font-size: 0.82rem;
        color: #6e7681;
        line-height: 1.5;
        font-family: 'Inter', sans-serif;
    }

    .alert-box {
        border: 1px solid #3d2f1d;
        background-color: #17130d;
        color: #d29922;
        padding: 1rem 1.25rem;
        border-radius: 6px;
        font-size: 0.88rem;
        margin-bottom: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Icono SVG (i) minimalista idéntico a la referencia visual
INFO_ICON_SVG = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" '
    'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="10"></circle>'
    '<line x1="12" y1="16" x2="12" y2="12"></line>'
    '<line x1="12" y1="8" x2="12.01" y2="8"></line>'
    '</svg>'
)


# 4. Caché para no recargar el modelo
@st.cache_resource
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    cadena = configurar_cadena_rag(vectorstore)
    return vectorstore, cadena


vectorstore, cadena_rag = cargar_sistema()

# Verificar cantidad de fragmentos indexados en ChromaDB
try:
    total_fragmentos = vectorstore._collection.count()
except Exception:
    total_fragmentos = 0

# 5. Inicializar historial de chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


def renderizar_fuentes(fuentes, pregunta_original="", rephrased_query="", tiempo_s=None, idioma="es"):
    """Renderiza el panel de referencias documentales con diseño sobrio y sin emojis."""
    if not fuentes and not rephrased_query:
        return
    textos = I18N[idioma]
    etiqueta = f"{textos['doc_references']} ({len(fuentes)})"
    if tiempo_s is not None:
        etiqueta += f"  ·  {tiempo_s:.1f}s"
    with st.expander(etiqueta):
        if rephrased_query and rephrased_query != pregunta_original:
            st.markdown(
                f"<div class='ref-rephrased'>{textos['contextualized_query']} \"{rephrased_query}\"</div>",
                unsafe_allow_html=True
            )
        for i, doc in enumerate(fuentes, start=1):
            fuente_raw = doc.get("source", "Desconocido") if isinstance(doc, dict) else doc.metadata.get("source", "Desconocido")
            nombre_archivo = os.path.basename(fuente_raw)
            pagina = doc.get("page", "N/A") if isinstance(doc, dict) else doc.metadata.get("page", "N/A")
            contenido = doc.get("page_content", "") if isinstance(doc, dict) else doc.page_content
            fragmento_corto = contenido[:320].replace("<", "&lt;").replace(">", "&gt;") + "..."
            st.markdown(
                f"""
                <div class="ref-card">
                    <div class="ref-header">
                        <span>REF {i:02d} — {nombre_archivo}</span>
                        <span>{textos['page_label']} {pagina}</span>
                    </div>
                    <div class="ref-excerpt">{fragmento_corto}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.markdown(f"<div class='sidebar-brand'>{t['sidebar_brand']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sidebar-title'>{t['sidebar_title']}</div>", unsafe_allow_html=True)

    if st.button(t["new_session"], use_container_width=True, type="secondary"):
        st.session_state.mensajes = []
        st.rerun()

    mensajes_activos = min(len(st.session_state.mensajes), MAX_HISTORY_MESSAGES)
    st.markdown(
        f"""
        <div class="meta-panel">
            <div class="meta-row">
                <span class="meta-label-group">
                    {t['llm_engine']}
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>{t['llm_tooltip_title']}</strong> {t['llm_tooltip_desc']}
                    </span>
                </span>
                <span class="meta-value">{t['llm_value']}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    {t['vector_index']}
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>{t['vector_tooltip_title']}</strong> {t['vector_tooltip_desc']}
                    </span>
                </span>
                <span class="meta-value">{total_fragmentos} {t['vectors_label']}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    {t['retrieval']}
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>{t['retrieval_tooltip_title']}</strong> {t['retrieval_tooltip_desc']}
                    </span>
                </span>
                <span class="meta-value">{t['retrieval_value']}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    {t['active_memory']}
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>{t['memory_tooltip_title']}</strong> {t['memory_tooltip_desc']}
                    </span>
                </span>
                <span class="meta-value">{mensajes_activos} / {MAX_HISTORY_MESSAGES} {t['messages_label']}</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    {t['temperature']}
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>{t['temp_tooltip_title']}</strong> {t['temp_tooltip_desc']}
                    </span>
                </span>
                <span class="meta-value">{t['temp_value']}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander(t["ingest_expander"]):
        archivos_subidos = st.file_uploader(
            t["ingest_uploader"],
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if archivos_subidos:
            if st.button(t["ingest_btn"], use_container_width=True):
                os.makedirs("data", exist_ok=True)
                total_nuevos = 0
                procesados = 0
                with st.spinner(t["ingest_processing"]):
                    for archivo in archivos_subidos:
                        try:
                            nombre_seguro = sanitizar_nombre_pdf(archivo.name)
                            contenido_bytes = archivo.getvalue()
                            if not contenido_bytes.startswith(b"%PDF-"):
                                st.error(t["ingest_invalid"].format(name=archivo.name))
                                continue
                            ruta_destino = os.path.join("data", nombre_seguro)
                            with open(ruta_destino, "wb") as f:
                                f.write(contenido_bytes)
                            num_chunks = procesar_y_agregar_pdf(ruta_destino, vectorstore)
                            total_nuevos += num_chunks
                            procesados += 1
                        except Exception as e:
                            st.error(t["ingest_error"].format(name=archivo.name, err=e))
                if procesados > 0:
                    st.success(t["ingest_success"].format(chunks=total_nuevos, docs=procesados))
                    st.rerun()

        if os.path.exists("data"):
            pdfs_actuales = sorted([f for f in os.listdir("data") if f.lower().endswith(".pdf")])
            if pdfs_actuales:
                st.caption(t["docs_in_repo"].format(count=len(pdfs_actuales)))
                for pdf_name in pdfs_actuales:
                    st.markdown(f"<div class='glossary-desc'>· {pdf_name}</div>", unsafe_allow_html=True)

    with st.expander(t["glossary_expander"]):
        for termino, definicion in t["glossary_items"]:
            st.markdown(
                f"""
                <div class="glossary-item">
                    <div class="glossary-term">{termino}</div>
                    <div class="glossary-desc">{definicion}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

# --- ÁREA PRINCIPAL (HEADER CON SELECTOR DE IDIOMA EN LA PARTE SUPERIOR) ---

col_top_brand, col_top_lang = st.columns([5, 1.8])
with col_top_brand:
    st.markdown(f"<div class='top-brand'>{t['top_brand']}</div>", unsafe_allow_html=True)
with col_top_lang:
    idioma_seleccionado = st.selectbox(
        "Language",
        options=["Español", "English"],
        index=0 if st.session_state.idioma == "es" else 1,
        label_visibility="collapsed",
        key="selector_idioma_top"
    )
    codigo_seleccionado = "es" if idioma_seleccionado == "Español" else "en"
    if st.session_state.idioma != codigo_seleccionado:
        st.session_state.idioma = codigo_seleccionado
        st.rerun()

if total_fragmentos == 0:
    st.markdown(
        f"<div class='alert-box'>"
        f"<strong>{t['empty_alert_title']}</strong> {t['empty_alert_desc']}"
        f"</div>",
        unsafe_allow_html=True
    )

if len(st.session_state.mensajes) == 0:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="empty-kicker">{t['empty_kicker']}</div>
            <div class="empty-heading">{t['empty_heading']}</div>
            <div class="empty-sub">{t['empty_sub']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Mostrar historial de conversación
for msj in st.session_state.mensajes:
    etiqueta_rol = t["role_user"] if msj["rol"] == "user" else t["role_assistant"]
    with st.chat_message(msj["rol"]):
        st.markdown(f"<div class='role-label'>{etiqueta_rol}</div>", unsafe_allow_html=True)
        st.markdown(msj["contenido"])
        if msj["rol"] == "assistant" and msj.get("fuentes"):
            renderizar_fuentes(
                msj["fuentes"],
                msj.get("pregunta_original", ""),
                msj.get("rephrased_query", ""),
                msj.get("tiempo_s"),
                idioma=msj.get("idioma", st.session_state.idioma)
            )

# Input del usuario con límite de caracteres (SEC-01)
pregunta = st.chat_input(
    t["chat_input_placeholder"],
    max_chars=MAX_QUESTION_LENGTH
)

if pregunta and pregunta.strip():
    pregunta_limpia = pregunta.strip()
    idioma_actual = st.session_state.idioma

    # 1. Guardar y mostrar la pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_limpia, "idioma": idioma_actual})
    with st.chat_message("user"):
        st.markdown(f"<div class='role-label'>{t['role_user']}</div>", unsafe_allow_html=True)
        st.markdown(pregunta_limpia)

    # 2. Reconstruir el historial aplicando ventana deslizante (SEC-01)
    chat_history = []
    for msg in st.session_state.mensajes[:-1][-MAX_HISTORY_MESSAGES:]:
        if msg["rol"] == "user":
            chat_history.append(HumanMessage(content=msg["contenido"]))
        elif msg["rol"] == "assistant":
            chat_history.append(AIMessage(content=msg["contenido"]))

    # 3. Generar respuesta en Streaming en tiempo real
    with st.chat_message("assistant"):
        st.markdown(f"<div class='role-label'>{t['role_assistant']}</div>", unsafe_allow_html=True)
        inicio_t = time.perf_counter()
        try:
            with st.spinner(t["analyzing_spinner"]):
                resultado = cadena_rag.invoke({
                    "question": pregunta_limpia,
                    "chat_history": chat_history,
                    "return_stream": True,
                    "idioma": idioma_actual
                })
                documentos_fuente = resultado["docs"]
                rephrased_query = resultado.get("rephrased_query", pregunta_limpia)

            respuesta = st.write_stream(resultado["stream"])
            tiempo_total = time.perf_counter() - inicio_t

            fuentes_serializadas = [
                {
                    "source": doc.metadata.get("source", "Desconocido"),
                    "page": doc.metadata.get("page", "N/A"),
                    "page_content": doc.page_content
                }
                for doc in documentos_fuente
            ]

            renderizar_fuentes(fuentes_serializadas, pregunta_limpia, rephrased_query, tiempo_total, idioma=idioma_actual)

            st.session_state.mensajes.append({
                "rol": "assistant",
                "contenido": respuesta,
                "fuentes": fuentes_serializadas,
                "pregunta_original": pregunta_limpia,
                "rephrased_query": rephrased_query,
                "tiempo_s": tiempo_total,
                "idioma": idioma_actual
            })
        except Exception as e:
            st.error(t["connection_error"])
            st.caption(f"Excepción / Exception: {type(e).__name__}")
