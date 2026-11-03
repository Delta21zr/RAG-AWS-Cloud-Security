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

# 1. Configuración de la página (Sin emojis, diseño sobrio y ejecutivo)
st.set_page_config(
    page_title="AWS Security — Documentation Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CSS minimalista con tooltips interactivos estilo (i) y glosario técnico
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
        padding-top: 2.5rem !important;
        padding-bottom: 6rem !important;
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

    /* Tooltip flotante al pasar el cursor sobre el icono (i) */
    .info-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        left: 0;
        right: 0;
        bottom: calc(100% + 6px);
        background-color: #181c24;
        color: #c9d1d9;
        border: 1px solid #2d333b;
        border-radius: 6px;
        padding: 0.65rem 0.75rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.74rem;
        font-weight: 400;
        line-height: 1.45;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.55);
        z-index: 999;
        pointer-events: none;
        transition: opacity 0.16s ease, transform 0.16s ease;
        transform: translateY(4px);
    }

    .info-trigger:hover + .info-tooltip {
        visibility: visible;
        opacity: 1;
        transform: translateY(0);
    }

    .meta-value {
        color: #c9d1d9;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
    }

    /* Glosario en barra lateral */
    .glossary-item {
        padding: 0.55rem 0;
        border-bottom: 1px solid #161920;
    }

    .glossary-item:last-child {
        border-bottom: none;
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
        margin-top: 14vh;
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


# 3. Caché para no recargar el modelo
@st.cache_resource
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    cadena = configurar_cadena_rag(vectorstore)
    return vectorstore, cadena


vectorstore, cadena_rag = cargar_sistema()

# Verificar cantidad de fragmentos indexados en ChromaDB (BUG-02)
try:
    total_fragmentos = vectorstore._collection.count()
except Exception:
    total_fragmentos = 0

# 4. Inicializar historial de chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []


def renderizar_fuentes(fuentes, pregunta_original="", rephrased_query="", tiempo_s=None):
    """Renderiza el panel de referencias documentales con diseño sobrio y sin emojis."""
    if not fuentes and not rephrased_query:
        return
    etiqueta = f"Referencias documentales ({len(fuentes)})"
    if tiempo_s is not None:
        etiqueta += f"  ·  {tiempo_s:.1f}s"
    with st.expander(etiqueta):
        if rephrased_query and rephrased_query != pregunta_original:
            st.markdown(
                f"<div class='ref-rephrased'>Consulta contextualizada: \"{rephrased_query}\"</div>",
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
                        <span>Página {pagina}</span>
                    </div>
                    <div class="ref-excerpt">{fragmento_corto}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>AWS Cloud Security</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>Documentation RAG</div>", unsafe_allow_html=True)

    if st.button("Nueva sesión", use_container_width=True, type="secondary"):
        st.session_state.mensajes = []
        st.rerun()

    mensajes_activos = min(len(st.session_state.mensajes), MAX_HISTORY_MESSAGES)
    st.markdown(
        f"""
        <div class="meta-panel">
            <div class="meta-row">
                <span class="meta-label-group">
                    Motor LLM
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>Large Language Model:</strong> Modelo de lenguaje ejecutado localmente mediante Ollama. Redacta las respuestas basándose en los fragmentos recuperados sin enviar datos a internet.
                    </span>
                </span>
                <span class="meta-value">Llama 3.1 (Local)</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    Índice vectorial
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>Base de Datos Vectorial (ChromaDB):</strong> Total de fragmentos de texto (chunks) extraídos de los PDFs y convertidos en vectores matemáticos multilingües para búsqueda semántica.
                    </span>
                </span>
                <span class="meta-value">{total_fragmentos} vectores</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    Recuperación
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>Búsqueda Híbrida + Cross-Encoder:</strong> Combina similitud vectorial (ChromaDB) con palabras clave exactas (BM25) mediante Reciprocal Rank Fusion y re-ordena los candidatos con un modelo Cross-Encoder multilingüe.
                    </span>
                </span>
                <span class="meta-value">Híbrida + Rerank</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    Memoria activa
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>Ventana Deslizante (Sliding Window):</strong> Cantidad de mensajes recientes que el sistema recuerda para entender preguntas de seguimiento, limitada a {MAX_HISTORY_MESSAGES} para proteger la memoria RAM.
                    </span>
                </span>
                <span class="meta-value">{mensajes_activos} / {MAX_HISTORY_MESSAGES} msjs</span>
            </div>
            <div class="meta-row">
                <span class="meta-label-group">
                    Temperatura
                    <span class="info-trigger">{INFO_ICON_SVG}</span>
                    <span class="info-tooltip">
                        <strong>Control de Creatividad:</strong> El valor 0.0 elimina la aleatoriedad del modelo, obligándolo a dar respuestas técnicas precisas, deterministas y sin alucinaciones.
                    </span>
                </span>
                <span class="meta-value">0.0 (Determinista)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.expander("Ingesta de documentos (PDF)"):
        archivos_subidos = st.file_uploader(
            "Seleccione uno o más archivos PDF",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if archivos_subidos:
            if st.button("Indexar en base de datos", use_container_width=True):
                os.makedirs("data", exist_ok=True)
                total_nuevos = 0
                procesados = 0
                with st.spinner("Procesando e indexando PDF(s)..."):
                    for archivo in archivos_subidos:
                        try:
                            nombre_seguro = sanitizar_nombre_pdf(archivo.name)
                            contenido_bytes = archivo.getvalue()
                            if not contenido_bytes.startswith(b"%PDF-"):
                                st.error(f"Archivo inválido ({archivo.name}): no es un PDF válido.")
                                continue
                            ruta_destino = os.path.join("data", nombre_seguro)
                            with open(ruta_destino, "wb") as f:
                                f.write(contenido_bytes)
                            num_chunks = procesar_y_agregar_pdf(ruta_destino, vectorstore)
                            total_nuevos += num_chunks
                            procesados += 1
                        except Exception as e:
                            st.error(f"Error al procesar {archivo.name}: {e}")
                if procesados > 0:
                    st.success(f"Se indexaron {total_nuevos} fragmentos de {procesados} documento(s).")
                    st.rerun()

        if os.path.exists("data"):
            pdfs_actuales = sorted([f for f in os.listdir("data") if f.lower().endswith(".pdf")])
            if pdfs_actuales:
                st.caption(f"Documentos en repositorio ({len(pdfs_actuales)}):")
                for pdf_name in pdfs_actuales:
                    st.markdown(f"<div class='glossary-desc'>· {pdf_name}</div>", unsafe_allow_html=True)

    with st.expander("Glosario de términos"):
        st.markdown(
            """
            <div class="glossary-item">
                <div class="glossary-term">RAG (Retrieval-Augmented Generation)</div>
                <div class="glossary-desc">Arquitectura que busca primero en tus documentos PDF antes de generar una respuesta con IA.</div>
            </div>
            <div class="glossary-item">
                <div class="glossary-term">Embeddings Multilingües</div>
                <div class="glossary-desc">Representación numérica del significado de un texto que permite buscar en español dentro de PDFs en inglés.</div>
            </div>
            <div class="glossary-item">
                <div class="glossary-term">IAM (Identity and Access Management)</div>
                <div class="glossary-desc">Servicio de AWS para controlar usuarios, roles, políticas y el principio de privilegio mínimo.</div>
            </div>
            <div class="glossary-item">
                <div class="glossary-term">Security Group vs. NACL</div>
                <div class="glossary-desc">Firewalls virtuales en AWS: Security Group opera a nivel de instancia EC2 (con estado / stateful) y NACL a nivel de subred (sin estado / stateless).</div>
            </div>
            <div class="glossary-item">
                <div class="glossary-term">CloudTrail / CloudWatch / GuardDuty</div>
                <div class="glossary-desc">Tríada de monitoreo: CloudTrail audita llamadas a la API, CloudWatch supervisa métricas/logs operativos y GuardDuty detecta amenazas con IA.</div>
            </div>
            <div class="glossary-item">
                <div class="glossary-term">CSE vs. SSE (Cifrado)</div>
                <div class="glossary-desc">Client-Side Encryption cifra los datos antes de subirlos a AWS; Server-Side Encryption los cifra automáticamente en los servidores de AWS.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# --- ÁREA PRINCIPAL DE CONSULTA ---

if total_fragmentos == 0:
    st.markdown(
        "<div class='alert-box'>"
        "<strong>Índice vectorial vacío.</strong> Ejecute <code>python ingestar_datos.py</code> "
        "en la terminal para procesar los documentos de la carpeta <code>data/</code>."
        "</div>",
        unsafe_allow_html=True
    )

if len(st.session_state.mensajes) == 0:
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-kicker">Sistema de Consulta Privado</div>
            <div class="empty-heading">Asistente Técnico de Seguridad en AWS</div>
            <div class="empty-sub">
                Realice consultas específicas sobre control de acceso (IAM), seguridad de red (VPC, Security Groups, NACLs),
                cifrado (KMS, S3), monitoreo (CloudTrail, CloudWatch, GuardDuty) y respuesta a incidentes.
                Las respuestas se generan exclusivamente a partir de la documentación local verificada.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Mostrar historial de conversación
for msj in st.session_state.mensajes:
    etiqueta_rol = "Consulta" if msj["rol"] == "user" else "Respuesta"
    with st.chat_message(msj["rol"]):
        st.markdown(f"<div class='role-label'>{etiqueta_rol}</div>", unsafe_allow_html=True)
        st.markdown(msj["contenido"])
        if msj["rol"] == "assistant" and msj.get("fuentes"):
            renderizar_fuentes(
                msj["fuentes"],
                msj.get("pregunta_original", ""),
                msj.get("rephrased_query", ""),
                msj.get("tiempo_s")
            )

# Input del usuario con límite de caracteres (SEC-01)
pregunta = st.chat_input(
    "Escriba su consulta técnica...",
    max_chars=MAX_QUESTION_LENGTH
)

if pregunta and pregunta.strip():
    pregunta_limpia = pregunta.strip()

    # 1. Guardar y mostrar la pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta_limpia})
    with st.chat_message("user"):
        st.markdown("<div class='role-label'>Consulta</div>", unsafe_allow_html=True)
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
        st.markdown("<div class='role-label'>Respuesta</div>", unsafe_allow_html=True)
        inicio_t = time.perf_counter()
        try:
            with st.spinner("Analizando documentación..."):
                resultado = cadena_rag.invoke({
                    "question": pregunta_limpia,
                    "chat_history": chat_history,
                    "return_stream": True
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

            renderizar_fuentes(fuentes_serializadas, pregunta_limpia, rephrased_query, tiempo_total)

            st.session_state.mensajes.append({
                "rol": "assistant",
                "contenido": respuesta,
                "fuentes": fuentes_serializadas,
                "pregunta_original": pregunta_limpia,
                "rephrased_query": rephrased_query,
                "tiempo_s": tiempo_total
            })
        except Exception as e:
            st.error(
                "Error de conexión con el motor de inferencia local (Ollama). "
                "Verifique que el servicio esté activo y ejecutando el modelo 'llama3.1'."
            )
            st.caption(f"Excepción: {type(e).__name__}")
