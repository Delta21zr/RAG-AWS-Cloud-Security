# AWS Cloud Security RAG (100% Local)

Un sistema **RAG (Retrieval-Augmented Generation) totalmente local y privado** diseñado para consultar documentos de seguridad en la nube de AWS. Construido con tecnología Open Source, sin dependencias de APIs de pago como OpenAI, asegurando la máxima privacidad de los datos.

## Características Principales
- **100% Local y Privado:** Utiliza **Ollama** con el modelo **Llama 3.1** y **ChromaDB** para la persistencia de vectores. Ningún dato sale de tu máquina.
- **Búsqueda Híbrida (Vectorial + BM25) y Re-Ranking (Cross-Encoder):** Combina recuperación semántica multilingüe (`paraphrase-multilingual-MiniLM-L12-v2`) con coincidencia léxica exacta (`IndiceBM25` + `Reciprocal Rank Fusion`) y re-ordenamiento neuronal (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) para capturar tanto conceptos generales como siglas técnicas poco frecuentes (`WORM`, `CloudHSM`, `WAF`).
- **Memoria Conversacional (History-Aware Retriever):** Capacidad de mantener el contexto en diálogos multi-turno con ventana deslizante (*Sliding Window*), utilizando reformulación de consultas (*Query Reformulation* con *Few-Shot Prompting*) para resolver referencias anafóricas antes de consultar la base de datos.
- **Guardrails Anti-alucinación y Seguridad OWASP LLM:** Prompt de sistema riguroso (Temperatura 0), aislamiento de contexto mediante delimitadores XML (`<contexto_documentos>`) contra *Indirect Prompt Injection* y validación/sanitización de entradas.
- **Ingesta Dinámica de PDFs en Caliente:** Carga e indexación de nuevos manuales PDF directamente desde la interfaz web con validación de firma mágica (`%PDF-`), prevención de *Path Traversal*, deduplicación de fragmentos existentes y reconstrucción automática del índice léxico BM25 sin reiniciar el servidor.
- **Evaluación Científica y Pruebas Automatizadas:** Evaluado metodológicamente con **Ragas** (*Context Precision*, *Context Recall*, *Faithfulness* y *Answer Relevancy*) e incluye una suite de 46 tests automatizados (`test_rag.py`).
- **Interfaz Sobria y Minimalista:** Interfaz construida con **Streamlit**, con transmisión de respuestas en vivo (*Streaming*), tooltips técnicos interactivos, glosario integrado, persistencia de fuentes consultadas y manejo de errores.

## Instalación y Uso

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

## Pruebas de Seguridad, Calidad y Evaluación (Ragas)

1. **Suite de Pruebas Unitarias y de Seguridad (`unittest`):**
   Valida el truncamiento de entradas, ventana deslizante de memoria, aislamiento XML contra *Prompt Injection* e integridad del dataset:
   ```bash
   python test_rag.py
   ```

2. **Evaluación Científica del Modelo (`Ragas`):**
   Evalúa el pipeline completo contra el *Ground Truth* en métricas de *Context Precision*, *Context Recall*, *Faithfulness* y *Answer Relevancy*:
   ```bash
   python evaluar_rag.py
   ```

## Stack Tecnológico
* **LLM:** Llama 3.1 (vía Ollama)
* **Embeddings:** HuggingFace (`paraphrase-multilingual-MiniLM-L12-v2`)
* **Vector Store:** ChromaDB
* **Orquestación:** LangChain
* **Evaluación:** Ragas
* **Frontend:** Streamlit
