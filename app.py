import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import configurar_cadena_rag

# 1. Configuración de la página (Tema oscuro y layout amplio)
st.set_page_config(page_title="ChatGPT Local", page_icon="🤖", layout="wide")

# CSS Personalizado para ocultar logos de Streamlit y dar estilo ChatGPT
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stChatFloatingInputContainer {padding-bottom: 20px;}
    .titulo-vacio {
        text-align: center; 
        margin-top: 15vh; 
        margin-bottom: 5vh; 
        font-size: 2.5rem; 
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# 2. Caché para no recargar el modelo
@st.cache_resource
def cargar_sistema():
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    cadena = configurar_cadena_rag(vectorstore)
    return vectorstore, cadena

vectorstore, cadena_rag = cargar_sistema()

# 3. Inicializar historial de chat
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

from langchain_core.messages import HumanMessage, AIMessage

# --- BARRA LATERAL (SIDEBAR) ESTILO CHATGPT ---
with st.sidebar:
    st.title("🤖 ChatGPT Local")
    if st.button("📝 Nuevo chat", use_container_width=True, type="secondary"):
        st.session_state.mensajes = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Estado de la Memoria")
    st.caption(f"🧠 Mensajes recordados: {len(st.session_state.mensajes)}")
    st.markdown("📍 AWS Cloud Security RAG")

# --- ÁREA PRINCIPAL DE CHAT ---

# Si no hay mensajes, mostrar el saludo centrado
if len(st.session_state.mensajes) == 0:
    st.markdown("<div class='titulo-vacio'>¿En qué piensas hoy?</div>", unsafe_allow_html=True)

# Mostrar mensajes anteriores
for msj in st.session_state.mensajes:
    avatar = "👤" if msj["rol"] == "user" else "🧠"
    with st.chat_message(msj["rol"], avatar=avatar):
        st.markdown(msj["contenido"])

# 4. Input del usuario
pregunta = st.chat_input("Pregunta lo que quieras sobre tus documentos...")

if pregunta:
    # 1. Guardar y mostrar la pregunta del usuario
    st.session_state.mensajes.append({"rol": "user", "contenido": pregunta})
    with st.chat_message("user", avatar="👤"):
        st.markdown(pregunta)

    # 2. Reconstruir el historial en el formato de LangChain (sin incluir la pregunta actual)
    chat_history = []
    for msg in st.session_state.mensajes[:-1]:
        if msg["rol"] == "user":
            chat_history.append(HumanMessage(content=msg["contenido"]))
        elif msg["rol"] == "assistant":
            chat_history.append(AIMessage(content=msg["contenido"]))

    # 3. Generar respuesta considerando el contexto previo
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Pensando y consultando documentos..."):
            # Ejecutar cadena RAG pasando la pregunta y el historial acumulado
            resultado = cadena_rag.invoke({
                "question": pregunta,
                "chat_history": chat_history,
                "return_docs": True
            })

            respuesta = resultado["answer"]
            documentos_fuente = resultado["docs"]
            rephrased_query = resultado.get("rephrased_query", pregunta)

            # Mostrar respuesta final
            st.markdown(respuesta)

            # Mostrar fuentes consultadas y si hubo reformulación por memoria
            with st.expander("🔍 Fuentes consultadas"):
                if rephrased_query != pregunta:
                    st.info(f"🔄 **Búsqueda adaptada con memoria:** *\"{rephrased_query}\"*")
                for i, doc in enumerate(documentos_fuente):
                    st.caption(f"**Fuente {i+1}:** `{doc.metadata.get('source', 'Desconocido')}` (Pág {doc.metadata.get('page', 'N/A')})")
                    st.text(doc.page_content[:300] + "...")

    # Guardar respuesta en historial
    st.session_state.mensajes.append({"rol": "assistant", "contenido": respuesta})