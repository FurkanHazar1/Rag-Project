# api_optimized.py - Performans optimize edilmiş API
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import uvicorn
import logging
import time
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache
import gc

# Hata kontrolü için
models_loaded = False
error_message = ""

try:
    from langchain_ollama import ChatOllama
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains.llm import LLMChain
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.prompts import PromptTemplate
    from translation import translate_tr_to_en, translate_en_to_tr
    from nlp import QueryCleaner
    imports_ok = True
except Exception as e:
    imports_ok = False
    error_message = f"Import hatası: {str(e)}"
    print(f"❌ {error_message}")

# Logging ayarları - performans için seviye düşürüldü
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = FastAPI(title="Okul Mevzuat Chat API", version="1.0.0")

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response modelleri
class ChatRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    conversation_id: str
    response_time: float
    success: bool
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str
    error_message: Optional[str] = None

# Global değişkenler
llm = None
retriever = None
combine_documents_chain = None
query_cleaner = None
executor = ThreadPoolExecutor(max_workers=4)  # Thread pool for blocking operations

# Cache'ler
@lru_cache(maxsize=100)
def cached_query_clean(query: str) -> str:
    """Query temizleme işlemini cache'ler"""
    if query_cleaner:
        return query_cleaner.clean_query(query)
    return query

# Translation cache
translation_cache = {}
cache_lock = threading.Lock()

def cached_translate_tr_to_en(text: str) -> str:
    """Çeviri işlemini cache'ler"""
    with cache_lock:
        if text in translation_cache:
            return translation_cache[text]
        
        result = translate_tr_to_en(text)
        if len(translation_cache) > 2000:
            with cache_lock:
                for _ in range(500):
                    if translation_cache:
                        translation_cache.popitem()
                
        translation_cache[text] = result
        return result

def cached_translate_en_to_tr(text: str) -> str:
    """Çeviri işlemini cache'ler"""
    with cache_lock:
        cache_key = f"en_to_tr_{text}"
        if cache_key in translation_cache:
            return translation_cache[cache_key]
        
        result = translate_en_to_tr(text)
        if len(translation_cache) > 2000:
            for _ in range(500):
                translation_cache.popitem()
        
        translation_cache[cache_key] = result
        return result

# Fonksiyonlar (sadece import başarılıysa)
if imports_ok:
    def load_llm():
        # Ollama için optimizasyon ayarları
        return ChatOllama(
            model="llama3.2", 
            temperature=0,
        )

    @lru_cache(maxsize=1)
    def get_faiss(model_name: str, index_path: str):
        """FAISS index'i cache'ler - bir kez yükler"""
        embedding = SentenceTransformerEmbeddings(
            model_name=model_name,
            cache_folder="./sentence_transformer_cache"  # Cache klasörü belirt
        )
        vector = FAISS.load_local(
            index_path, embedding, allow_dangerous_deserialization=True
        )
        return vector

    @lru_cache(maxsize=1)
    def get_prompt_template():
        template = """
        1. Use the following pieces of context to answer the question at the end.
        2. If you don't know the answer, just say that "I am sorry, I don't know".
        3. Keep the answer crisp and concise.
        
        Context: {context}
        Question: {question}

        Helpful Answer:
        """
        return ChatPromptTemplate.from_template(template)

    class OptimizedTranslatedStuffDocumentsChain(StuffDocumentsChain):
        def _get_inputs(self, docs, **kwargs):
            # Paralel çeviri işlemi
            translated_docs = []
            
            # Çeviri işlemlerini paralel yap
            with ThreadPoolExecutor(max_workers=3) as trans_executor:
                translation_futures = []
                for doc in docs:
                    future = trans_executor.submit(cached_translate_tr_to_en, doc.page_content)
                    translation_futures.append((doc, future))
                
                for doc, future in translation_futures:
                    translated_content = future.result()
                    translated_doc = type(doc)(
                        page_content=translated_content,
                        metadata=doc.metadata
                    )
                    translated_docs.append(translated_doc)

            if 'question' in kwargs:
                kwargs['question'] = cached_translate_tr_to_en(kwargs['question'])

            return super()._get_inputs(translated_docs, **kwargs)

    async def answer_turkish_question_async(turkish_question: str) -> str:
        """Asenkron soru cevaplama"""
        loop = asyncio.get_event_loop()
        
        # CPU-intensive işlemleri thread pool'da çalıştır
        def blocking_operation():
            # Query temizleme
            cleaned_query = cached_query_clean(turkish_question)
            
            # Doküman arama - en alakalı 3 doküman al (5 yerine)
            retrieved_docs = retriever.get_relevant_documents(cleaned_query)[:5]
            
            # Chain çalıştırma
            result = combine_documents_chain.run(
                input_documents=retrieved_docs,
                question=turkish_question
            )
            
            # Çeviri
            response_clear = cached_translate_en_to_tr(result)
            return response_clear
        
        # Blocking işlemi thread pool'da çalıştır
        result = await loop.run_in_executor(executor, blocking_operation)
        return result

# Startup işlemi - optimize edilmiş
@app.on_event("startup")
async def startup_event():
    global llm, retriever, combine_documents_chain, query_cleaner, models_loaded, error_message
    
    if not imports_ok:
        logger.error("❌ Import'lar başarısız, modeller yüklenmeyecek")
        return
    
    try:
        print("🔄 Dosya kontrolü yapılıyor...")
        
        # Dosya kontrolleri
        model_name = 'intfloat/multilingual-e5-base'
        faiss_index_path = 'faiss_index'
        
        if not os.path.exists(faiss_index_path):
            raise Exception(f"FAISS index klasörü bulunamadı: {faiss_index_path}")
        
        print("📦 Modeller yükleniyor...")
        
        # Paralel model yükleme
        async def load_models_parallel():
            loop = asyncio.get_event_loop()
            
            # LLM yükleme
            llm_future = loop.run_in_executor(executor, load_llm)
            
            # FAISS yükleme
            def load_faiss():
                vector = get_faiss(model_name, faiss_index_path)
                return vector.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 5}  # 5'ten 3'e düşürüldü
                )
            
            faiss_future = loop.run_in_executor(executor, load_faiss)
            
            # QueryCleaner yükleme
            def load_cleaner():
                return QueryCleaner()
            
            cleaner_future = loop.run_in_executor(executor, load_cleaner)
            
            # Tümünü bekle
            llm_result, retriever_result, cleaner_result = await asyncio.gather(
                llm_future, faiss_future, cleaner_future
            )
            
            return llm_result, retriever_result, cleaner_result
        
        llm, retriever, query_cleaner = await load_models_parallel()
        
        # Zincirler - optimize edilmiş
        prompts = get_prompt_template()
        llm_chain = LLMChain(llm=llm, prompt=prompts, verbose=False)
        
        document_prompt = PromptTemplate(
            input_variables=["page_content", "source"],
            template="Context:\ncontent:{page_content}",  # source kaldırıldı
        )
        
        combine_documents_chain = OptimizedTranslatedStuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
        )
        
        models_loaded = True
        print("✅ Tüm modeller başarıyla yüklendi!")
        
        # İlk warm-up sorgusu
        try:
            await answer_turkish_question_async("test")
            print("✅ Warm-up tamamlandı")
        except:
            pass
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"❌ Model yükleme hatası: {error_message}")
        models_loaded = False

# Memory cleanup task
async def cleanup_memory():
    """Periyodik olarak memory temizliği yapar"""
    while True:
        await asyncio.sleep(300)  # 5 dakikada bir
        gc.collect()
        # Cache boyutunu kontrol et
        if len(translation_cache) > 300:
            with cache_lock:
                for _ in range(100):
                    if translation_cache:
                        translation_cache.popitem()

@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_memory())

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Okul Mevzuat Chat API", "docs_url": "/docs", "models_loaded": models_loaded}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if models_loaded else "models_not_loaded",
        models_loaded=models_loaded,
        version="1.0.0",
        error_message=error_message if error_message else None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Soru boş olamaz")
    
    if not models_loaded:
        return ChatResponse(
            answer="Üzgünüm, AI modelleri henüz yüklenmedi veya bir hata oluştu. Lütfen sistem yöneticisine başvurun.",
            conversation_id=request.conversation_id or "error",
            response_time=0,
            success=False,
            error_message=error_message
        )
    
    try:
        start_time = time.time()
        
        # Asenkron soru cevaplama
        answer = await answer_turkish_question_async(request.question)
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            answer=answer,
            conversation_id=request.conversation_id or "default",
            response_time=round(response_time, 2),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Chat hatası: {str(e)}")
        return ChatResponse(
            answer="Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin.",
            conversation_id=request.conversation_id or "default",
            response_time=0,
            success=False,
            error_message=str(e)
        )

# Batch processing endpoint (opsiyonel)
@app.post("/chat/batch")
async def chat_batch_endpoint(requests: list[ChatRequest]):
    """Birden fazla soruyu paralel işler"""
    if not models_loaded:
        return {"error": "Models not loaded"}
    
    async def process_single_request(req):
        try:
            start_time = time.time()
            answer = await answer_turkish_question_async(req.question)
            response_time = time.time() - start_time
            
            return ChatResponse(
                answer=answer,
                conversation_id=req.conversation_id or "default",
                response_time=round(response_time, 2),
                success=True
            )
        except Exception as e:
            return ChatResponse(
                answer="Hata oluştu.",
                conversation_id=req.conversation_id or "default",
                response_time=0,
                success=False,
                error_message=str(e)
            )
    
    # Paralel işleme
    results = await asyncio.gather(*[process_single_request(req) for req in requests[:5]])  # Max 5 soru
    return {"results": results}

if __name__ == "__main__":
    print("🚀 Optimize edilmiş API başlatılıyor...")
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000, 
        reload=False,
        workers=1,  # Tek worker, async kullanıyoruz
        loop="asyncio",
        access_log=False  # Access log'u kapat
    )