import os
import json
import pandas as pd
from datasets import Dataset
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Importamos tu configuración real del RAG
from rag_pipeline import configurar_cadena_rag

def cargar_dataset(ruta_json="dataset_evaluacion.json"):
    print(f"Cargando preguntas de prueba desde {ruta_json}...")
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos = json.load(f)
    
    preguntas = [item["question"] for item in datos]
    respuestas_ideales = [item["ground_truth"] for item in datos]
    return preguntas, respuestas_ideales

def ejecutar_evaluacion():
    # 1. Cargar el dataset (Ground Truth)
    preguntas_prueba, respuestas_ideales = cargar_dataset()

    print("1. Cargando sistema RAG local...")
    embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    vectorstore = Chroma(persist_directory="db", embedding_function=embeddings)
    cadena_rag = configurar_cadena_rag(vectorstore)
    
    # Creamos un retriever exactamente igual al que usas en producción
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("2. Generando respuestas y recuperando contextos del RAG...")
    data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for i, pregunta in enumerate(preguntas_prueba):
        print(f"\nProcesando pregunta [{i+1}/{len(preguntas_prueba)}]: {pregunta}")
        
        # Ejecutar tu RAG para obtener la respuesta de Llama 3.1
        respuesta_generada = cadena_rag.invoke(pregunta)
        
        # Ejecutar el retriever para obtener qué trozos de texto usó (Contextos)
        documentos_recuperados = retriever.invoke(pregunta)
        contextos = [doc.page_content for doc in documentos_recuperados]

        # Guardamos en la estructura requerida por Ragas
        data["question"].append(pregunta)
        data["answer"].append(respuesta_generada)
        data["contexts"].append(contextos)
        data["ground_truth"].append(respuestas_ideales[i])

    # Convertir a formato Dataset de HuggingFace (Requerido por Ragas)
    dataset = Dataset.from_dict(data)

    print("\n3. Iniciando Evaluación con Ragas (Usando Llama 3.1 como Juez)...")
    print("Esto puede tardar unos minutos ya que el LLM evaluará cada respuesta.")
    
    # Instanciamos el modelo local que actuará como juez evaluador
    juez_llm = ChatOllama(model="llama3.1", temperature=0)
    
    # Ragas usa embeddings internamente para calcular similitudes
    juez_embeddings = HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

    from ragas.run_config import RunConfig
    
    # Ejecutar evaluación
    resultados = evaluate(
        dataset=dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        llm=juez_llm,
        embeddings=juez_embeddings,
        run_config=RunConfig(timeout=600, max_workers=1)
    )

    print("\n========= RESULTADOS DE LA EVALUACIÓN =========")
    df_resultados = resultados.to_pandas()
    
    # Mostrar por consola las primeras filas con todas sus columnas
    print(df_resultados.head())
    
    # Guardar reporte en CSV
    df_resultados.to_csv("reporte_evaluacion_rag.csv", index=False)
    print("\n✅ Reporte completo guardado en 'reporte_evaluacion_rag.csv'")
    print("Abre este CSV en Excel para analizar en qué preguntas falló el RAG.")

if __name__ == "__main__":
    ejecutar_evaluacion()