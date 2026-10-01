import os
import shutil
from rag_pipeline import cargar_y_dividir_documentos, crear_o_cargar_vectorstore

def reingestar_todo():
    # 1. Borrar la base de datos antigua
    if os.path.exists("db"):
        print("🗑️ Borrando la base de datos antigua (carpeta db/)...")
        shutil.rmtree("db")
    
    # 2. Cargar y dividir los PDFs de nuevo (ahora usará tu nuevo chunk_size)
    print("📄 Leyendo PDFs de la carpeta data/...")
    fragmentos = cargar_y_dividir_documentos("data")
    
    # 3. Crear y guardar la nueva base de datos Vectorial
    if fragmentos:
        print("🧠 Creando nueva base de datos con los nuevos fragmentos...")
        crear_o_cargar_vectorstore(fragmentos, "db")
        print("✅ ¡Proceso terminado con éxito! Tu base de datos ha sido actualizada.")
    else:
        print("❌ Error: No se encontraron documentos o no se pudieron procesar.")

if __name__ == "__main__":
    reingestar_todo()
