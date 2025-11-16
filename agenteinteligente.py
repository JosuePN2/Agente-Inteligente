import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader

import speech_recognition as sr
from gtts import gTTS
import playsound
import textwrap

# CONFIG
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  
MANUAL_PATH = "manual.pdf"  
TOP_K = 3  

# Voz
def ouvir_comando():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Fale agora...")
        r.adjust_for_ambient_noise(source, duration=0.6)
        audio = r.listen(source, timeout=6, phrase_time_limit=10)
    try:
        texto = r.recognize_google(audio, language="pt-BR")
        print("Você disse:", texto)
        return texto
    except sr.UnknownValueError:
        print("Não entendi.")
        return ""
    except sr.RequestError as e:
        print("Erro no serviço de reconhecimento:", e)
        return ""

import uuid
import os

def falar(texto):
    if not texto:
        return

    print(f"Assistente: {texto}")

    filename = f"resposta_{uuid.uuid4().hex[:8]}.mp3"

    try:
        tts = gTTS(text=texto, lang="pt-br")
        tts.save(filename)
        playsound.playsound(filename)
    finally:
        # Remove o arquivo temporário após tocar
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                pass  

# Carregar e indexar manual
import os

def carregar_e_indexar(path_manual):
    print("Carregando manual:", path_manual)
    
    extensao = os.path.splitext(path_manual)[-1].lower()
    loader = None

    if extensao == ".txt":
        loader = TextLoader(path_manual, encoding="utf-8")
    elif extensao == ".pdf":
        loader = PyPDFLoader(path_manual)
    elif extensao in [".docx", ".doc"]:
        loader = UnstructuredWordDocumentLoader(path_manual)
    else:
        print(f"Erro: Formato de arquivo '{extensao}' não suportado.")
        return None 

    try:
        docs = loader.load() 
    except Exception as e:
        print(f"Erro ao carregar o arquivo {path_manual}: {e}")
        return None
        
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"Criados {len(chunks)} chunks.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    print("Index criado (FAISS).")
    return vectorstore

# montar prompt
def montar_prompt(context_chunks, pergunta):

    context_text = "\n\n---\n\n".join([c.page_content for c in context_chunks])
    system_prompt = (
        "Você é um assistente técnico que responde apenas com base no manual fornecido. "
        "Se a resposta não estiver no manual, diga honestamente que não encontrou a informação. "
        "Seja conciso e prático."
    )
    prompt = f"{system_prompt}\n\nContexto:\n{context_text}\n\nPergunta: {pergunta}\nResposta:"
    return prompt

#import google.generativeai as genai

llm_cache = None

def gerar_resposta_via_gemini(prompt):
    global llm_cache
    try:
        if llm_cache is None:
            print("Iniciando modelo Gemini (primeira chamada)...")
            llm_cache = ChatGoogleGenerativeAI(model="gemini-2.5-flash", 
                                             google_api_key=GOOGLE_API_KEY,
                                             temperature=0.3) 
        response = llm_cache.invoke(prompt)
        return response.content

    except Exception as e:
        print("Erro ao chamar o Gemini (via LangChain):", e)
        return "Desculpe, não consegui gerar a resposta (erro no Gemini)."


# Loop principal
def executar_assistente(path_manual):
    vectorstore = carregar_e_indexar(path_manual)
    if vectorstore is None:
        print("Assistente encerrado devido a erro na indexação.")
        return 

    print("Assistente pronto. Diga 'sair' para encerrar.")
    while True:
        pergunta = ouvir_comando()
        if not pergunta:
            continue
        if "sair" in pergunta.lower() or "parar" in pergunta.lower():
            falar("Encerrando o assistente. Até logo!")
            break
        try:
            top_docs = vectorstore.similarity_search(pergunta, k=TOP_K)
        except Exception as e:
            print("Erro na busca vetorial:", e)
            top_docs = []
        prompt = montar_prompt(top_docs, pergunta)
        resposta = gerar_resposta_via_gemini(prompt)
        falar(resposta)

if __name__ == "__main__":
    executar_assistente(MANUAL_PATH)
