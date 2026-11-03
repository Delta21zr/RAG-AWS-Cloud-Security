import os
import shutil
import sys
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rag_pipeline import cargar_y_dividir_documentos, crear_o_cargar_vectorstore


def reingestar_todo():
    # 1. Intentar borrar la carpeta db/, o vaciar la colección SQLite en caliente si Streamlit está abierto (BUG-03)
    if os.path.exists("db"):
        print("🗑️ Limpiando la base de datos antigua (carpeta db/)...")
        try:
            shutil.rmtree("db")
        except PermissionError:
            print("ℹ️ Streamlit tiene abierto el archivo SQLite. Limpiando la colección en caliente...")
            try:
                embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
                vs_existente = Chroma(persist_directory="db", embedding_function=embeddings)
                vs_existente.delete_collection()
                print("✅ Colección anterior vaciada correctamente sin necesidad de cerrar Streamlit.")
            except Exception as e:
                print(
                    f"\n❌ No se pudo limpiar la base de datos ({type(e).__name__}).\n"
                    "👉 Por favor, detén Streamlit (Ctrl + C) antes de ejecutar este script."
                )
                sys.exit(1)
    
    # 2. Cargar, limpiar y dividir los PDFs de nuevo
    print("📄 Leyendo y limpiando PDFs de la carpeta data/...")
    fragmentos = cargar_y_dividir_documentos("data")
    
    # 3. Crear y guardar la nueva base de datos Vectorial
    if fragmentos:
        print("🧠 Indexando nuevos fragmentos limpios en ChromaDB...")
        crear_o_cargar_vectorstore(fragmentos, "db")
        print("✅ ¡Proceso terminado con éxito! Tu base de datos ha sido actualizada.")
    else:
        print("❌ Error: No se encontraron documentos o no se pudieron procesar.")


if __name__ == "__main__":
    reingestar_todo()
