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

def configurar_cadena_rag(vectorstore):
    print("\nConectando con Ollama (Llama 3.1)...")
    # Temperature 0 es vital para RAG: queremos precisión, no creatividad.
    llm = ChatOllama(model="llama3.1", temperature=0)
    
    # Configuramos la BD como un "recuperador" de los 3 mejores fragmentos
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    # GUARDRAIL: System prompt estricto anti-alucinaciones
    template = """Actúa como un asistente técnico experto.
    Utiliza ÚNICAMENTE el siguiente contexto para responder a la pregunta. 
    Si la respuesta no se encuentra en el contexto, responde EXACTAMENTE: "No tengo información suficiente en los documentos proporcionados para responder a esta pregunta."
    No inventes información, no uses tus conocimientos previos y no alucines.

    Contexto:
    {context}

    Pregunta: {question}
    
    Respuesta:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Función auxiliar para unir los textos recuperados
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Magia de LCEL: Construimos la tubería (Pipeline)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

if __name__ == "__main__":
    print("Cargando BD Vectorial desde el disco...")
    # Cargamos directamente desde el disco para ahorrar tiempo
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    
    # Inicializamos la Cadena LCEL
    cadena_rag = configurar_cadena_rag(vectorstore)
    
    print("\n=============================================")
    print("PRUEBA 1: Pregunta TÉCNICA (En Español con Modelo Multilingüe)")
    pregunta_1 = "¿Qué es EC2 y quién gestiona su sistema operativo?"
    print(f"Usuario: {pregunta_1}")
    
    # Vamos a ver qué recupera ChromaDB internamente
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(pregunta_1)
    print(f"\n--- Lo que ChromaDB encontró (Mejor coincidencia) ---")
    if docs:
        print(f"Fuente: {docs[0].metadata['source']}, Pág: {docs[0].metadata.get('page')}")
        print(f"Texto: {docs[0].page_content[:200]}...\n-----------------------------------------------------")
    
    respuesta_1 = cadena_rag.invoke(pregunta_1)
    print(f"\nLlama 3.1: {respuesta_1}")
    
    print("\n=============================================")
    print("PRUEBA 2: Prueba de GUARDRAIL (Anti-alucinación)")
    pregunta_2 = "¿Cuál es la receta secreta para hacer un pastel de chocolate?"
    print(f"Usuario: {pregunta_2}")
    respuesta_2 = cadena_rag.invoke(pregunta_2)
    print(f"Llama 3.1: {respuesta_2}")
    print("=============================================")