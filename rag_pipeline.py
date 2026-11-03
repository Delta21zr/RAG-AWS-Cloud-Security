import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def cargar_y_dividir_documentos(directorio_data="data"):
    print(f"Cargando PDFs desde la carpeta '{directorio_data}'...")
    
    # Cargamos todos los PDFs del directorio
    loader = PyPDFDirectoryLoader(directorio_data)
    documentos = loader.load()
    
    if not documentos:
        print("⚠️ No se encontraron documentos en la carpeta data/.")
        return []
        
    print(f"✅ Se cargaron {len(documentos)} páginas. Dividiendo en fragmentos (chunks)...")
    
    # Parámetros optimizados para RAM/VRAM (chunks pequeños)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""] # Intenta separar por párrafos primero
    )
    
    fragmentos = text_splitter.split_documents(documentos)
    print(f"✅ Se generaron {len(fragmentos)} fragmentos en total.")
    
    return fragmentos

def crear_o_cargar_vectorstore(fragmentos, directorio_db="db"):
    print("\nIniciando motor de Embeddings (paraphrase-multilingual-MiniLM-L12-v2)...")
    # Este modelo es ligero y se ejecutará súper rápido en tu CPU o GPU
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    print(f"Indexando {len(fragmentos)} fragmentos en ChromaDB (esto puede tardar unos segundos la primera vez)...")
    # Guardamos los vectores en la carpeta db/ para que sea persistente
    vectorstore = Chroma.from_documents(
        documents=fragmentos, 
        embedding=embeddings, 
        persist_directory=directorio_db
    )
    print("✅ Base de datos vectorial creada y guardada en disco.")
    return vectorstore

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableLambda

def configurar_cadena_rag(vectorstore):
    print("\nConectando con Ollama (Llama 3.1)...")
    # Temperature 0 es vital para RAG: queremos precisión, no creatividad.
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    # Configuramos la BD como un "recuperador" de los 5 mejores fragmentos
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # 1. Cadena de Reformulación de Preguntas (History-Aware con Few-Shot)
    rephrase_system_prompt = (
        "Dado un historial de conversación y la última pregunta del usuario, reformula la pregunta para que sea "
        "una consulta de búsqueda directa e independiente en una base de datos vectorial.\n\n"
        "REGLAS CRÍTICAS:\n"
        "1. Conserva SIEMPRE el nombre del servicio o herramienta de AWS de la que se viene hablando "
        "(ej. 'Amazon GuardDuty', 'EC2', 'KMS', 'S3'). NUNCA lo omitas ni lo reemplaces por generalidades como 'la seguridad en la nube'.\n"
        "2. Devuelve SIEMPRE una PREGUNTA, NUNCA una respuesta ni afirmaciones.\n"
        "3. Devuelve ÚNICAMENTE la pregunta reformulada, sin explicaciones ni texto introductorio.\n\n"
        "Ejemplo 1:\n"
        "Historial: Usuario pregunta '¿Qué es Amazon GuardDuty?' y el Asistente responde.\n"
        "Pregunta actual: '¿Qué ejemplos de actividades maliciosas puede detectar en instancias EC2?'\n"
        "Pregunta reformulada: '¿Qué ejemplos de actividades maliciosas puede detectar Amazon GuardDuty en instancias EC2?'\n\n"
        "Ejemplo 2:\n"
        "Historial: Usuario pregunta '¿Qué es AWS KMS?' y el Asistente responde.\n"
        "Pregunta actual: '¿Cuáles son sus ventajas?'\n"
        "Pregunta reformulada: '¿Cuáles son las ventajas de AWS KMS?'"
    )
    rephrase_prompt = ChatPromptTemplate.from_messages([
        ("system", rephrase_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    # 2. Prompt técnico con GUARDRAIL anti-alucinaciones
    qa_system_prompt = (
        "Actúa como un asistente técnico experto en AWS Cloud Security.\n"
        "Utiliza ÚNICAMENTE el siguiente contexto para responder a la pregunta.\n"
        "Si la respuesta no se encuentra en el contexto, responde EXACTAMENTE: "
        "'No tengo información suficiente en los documentos proporcionados para responder a esta pregunta.'\n"
        "No inventes información, no uses tus conocimientos previos y no alucines.\n\n"
        "Contexto:\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    
    # Función auxiliar para unir los textos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Pipeline ejecutor que maneja preguntas directas o con historial
    def ejecutar_pipeline(input_data):
        if isinstance(input_data, str):
            question = input_data
            chat_history = []
        else:
            question = input_data.get("question", "")
            chat_history = input_data.get("chat_history", [])

        # Si hay historial previo, reformulamos la pregunta para ChromaDB
        if chat_history:
            search_query = rephrase_chain.invoke({
                "chat_history": chat_history,
                "question": question
            })
            print(f"--> [MEMORIA] Pregunta reformulada para ChromaDB: '{search_query}'")
        else:
            search_query = question

        # Recuperamos los documentos pertinentes
        docs = retriever.invoke(search_query)
        context = format_docs(docs)

        # Generamos la respuesta final
        respuesta = (qa_prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "chat_history": chat_history,
            "question": question
        })

        if isinstance(input_data, dict) and input_data.get("return_docs", False):
            return {
                "answer": respuesta,
                "docs": docs,
                "rephrased_query": search_query
            }
        return respuesta
    
    return RunnableLambda(ejecutar_pipeline)

if __name__ == "__main__":
    print("Cargando BD Vectorial desde el disco...")
    # Cargamos directamente desde el disco para ahorrar tiempo
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    
    # Inicializamos la Cadena RAG con Memoria
    cadena_rag = configurar_cadena_rag(vectorstore)
    
    print("\n=============================================")
    print("TURNO 1: Pregunta directa")
    pregunta_1 = "¿Qué es EC2 y quién gestiona su sistema operativo?"
    print(f"Usuario: {pregunta_1}")
    respuesta_1 = cadena_rag.invoke(pregunta_1)
    print(f"\nLlama 3.1: {respuesta_1}")
    
    print("\n=============================================")
    print("TURNO 2: Pregunta de seguimiento (REFERENCIA ANAFÓRICA: 'sus beneficios')")
    pregunta_2 = "¿Cuáles son sus beneficios?"
    print(f"Usuario: {pregunta_2}")
    
    # Simulamos el historial del turno anterior
    historial = [
        HumanMessage(content=pregunta_1),
        AIMessage(content=respuesta_1)
    ]
    
    respuesta_2 = cadena_rag.invoke({
        "question": pregunta_2,
        "chat_history": historial
    })
    print(f"\nLlama 3.1: {respuesta_2}")
    
    print("\n=============================================")
    print("TURNO 3: Prueba de GUARDRAIL (Anti-alucinación)")
    pregunta_3 = "¿Cuál es la receta secreta para hacer un pastel de chocolate?"
    print(f"Usuario: {pregunta_3}")
    respuesta_3 = cadena_rag.invoke(pregunta_3)
    print(f"\nLlama 3.1: {respuesta_3}")
    print("=============================================")