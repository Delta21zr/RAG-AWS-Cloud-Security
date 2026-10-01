# 🛡️ AWS Cloud Security RAG (100% Local)

Un sistema **RAG (Retrieval-Augmented Generation) totalmente local y privado** diseñado para consultar documentos de seguridad en la nube de AWS. Construido con tecnología Open Source, sin dependencias de APIs de pago como OpenAI, asegurando la máxima privacidad de los datos.

## ✨ Características Principales
- **100% Local y Privado:** Utiliza **Ollama** con el modelo **Llama 3.1** y **ChromaDB** para la persistencia de vectores. Ningún dato sale de tu máquina.
- **Soporte Multilingüe (Cross-lingual RAG):** Gracias al modelo de embeddings `paraphrase-multilingual-MiniLM-L12-v2`, puedes consultar en **español** documentos que están escritos en **inglés** y obtener respuestas precisas en español.
- **Guardrails Anti-alucinación:** Prompt de sistema riguroso (Temperatura 0) diseñado para forzar al modelo a responder "No tengo información suficiente..." si la respuesta no se encuentra en el contexto proporcionado.
- **Evaluación Científica:** Evaluado metodológicamente utilizando el framework **Ragas** para asegurar altas métricas de Fidelidad (*Faithfulness*) y Relevancia de respuesta.
- **Interfaz estilo ChatGPT:** Interfaz limpia y responsiva construida con **Streamlit**.

## 🚀 Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Delta21zr/RAG-AWS-Cloud-Security.git
   cd RAG-AWS-Cloud-Security
   ```

2. **Crear y activar el entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   # En Mac/Linux:
   source venv/bin/activate
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Preparar la base de datos:**
   Coloca tus documentos en PDF dentro de la carpeta `data/` y luego ejecuta el script de ingesta para crear la base de datos vectorial local (ChromaDB):
   ```bash
   python ingestar_datos.py
   ```

5. **Iniciar la aplicación web:**
   Asegúrate de tener [Ollama](https://ollama.com/) instalado y corriendo con el modelo `llama3.1`.
   ```bash
   streamlit run app.py
   ```

## 🧪 Evaluación del Modelo (Ragas)
Este proyecto incluye un script de evaluación (`evaluar_rag.py`) que genera un dataset (*Ground Truth*) y evalúa el pipeline contra métricas de *Context Precision*, *Context Recall*, *Faithfulness*, y *Answer Relevancy*.

```bash
python evaluar_rag.py
```

## 🛠️ Stack Tecnológico
*   **LLM:** Llama 3.1 (vía Ollama)
*   **Embeddings:** HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)
*   **Vector Store:** ChromaDB
*   **Orquestación:** LangChain
*   **Evaluación:** Ragas
*   **Frontend:** Streamlit
