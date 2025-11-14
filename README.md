# Sistema RAG Avanzado con LangChain

**Autora:** Alison Geraldine Valderrama Munar  
**Curso:** AREP - Arquitecturas Empresariales  
**Universidad:** Escuela Colombiana de Ingeniería Julio Garavito

## Descripción General

Este proyecto implementa un sistema de **Retrieval-Augmented Generation (RAG)** de última generación que permite realizar consultas inteligentes y contextuales sobre documentación web especializada. El sistema combina la potencia de los Large Language Models (LLMs) de OpenAI con búsqueda semántica avanzada en bases de datos vectoriales para proporcionar respuestas precisas, fundamentadas y contextualizadas en tiempo real.

### ¿Qué es RAG?

RAG (Retrieval-Augmented Generation) es una arquitectura avanzada de IA que potencia las capacidades de los modelos de lenguaje al integrar un sistema de recuperación de información externa antes de generar respuestas. Esta técnica híbrida:

- **Reduce significativamente las alucinaciones** del modelo mediante anclaje en datos verificables
- **Permite trabajar con información actualizada** sin necesidad de reentrenamiento costoso del modelo
- **Mejora la precisión y relevancia** de las respuestas mediante búsqueda semántica contextual
- **Facilita la trazabilidad** al proporcionar las fuentes de información utilizadas

## Stack Tecnológico

- **[LangChain](https://www.langchain.com/)**: Framework orquestador para desarrollo de aplicaciones avanzadas con LLMs
- **[OpenAI GPT-4o](https://openai.com/)**: Modelo de lenguaje de última generación para comprensión y generación de texto
- **[OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)**: text-embedding-3-small (512 dimensiones) para representación vectorial semántica
- **[Pinecone](https://www.pinecone.io/)**: Base de datos vectorial serverless de alto rendimiento para búsqueda semántica escalable
- **[BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/)**: Librería robusta para parsing y extracción de contenido HTML
- **[python-dotenv](https://pypi.org/project/python-dotenv/)**: Gestión segura de variables de entorno y credenciales
- **Python 3.10+**: Lenguaje de programación principal con soporte completo para async/await

## Arquitectura del Sistema

```
┌──────────────────────┐
│   Fuente de Datos    │
│   (Documentos Web)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   WebBaseLoader      │ ──► Extracción selectiva de contenido HTML
│   + BeautifulSoup    │     (título, headers, contenido principal)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ RecursiveCharacter   │ ──► Fragmentación inteligente
│    TextSplitter      │     • Chunks: 1000 caracteres
└──────────┬───────────┘     • Overlap: 200 caracteres
           │
           ▼
┌──────────────────────┐
│  OpenAI Embeddings   │ ──► Vectorización semántica
│ text-embedding-3     │     • Dimensión: 512
│      -small          │     • Modelo: small
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   Pinecone Index     │ ──► Almacenamiento vectorial
│   (arep-taller)      │     • Métrica: cosine
└──────────┬───────────┘     • Tipo: serverless
           │
           ▼
┌──────────────────────┐
│    RAG Agent         │ ──► Motor de recuperación y generación
│   GPT-4o + Tools     │     1. Búsqueda semántica (top-k=2)
└──────────┬───────────┘     2. Construcción de contexto
           │                 3. Generación de respuesta
           ▼                 4. Streaming en tiempo real
┌──────────────────────┐
│ Respuesta Streaming  │
│   Contextualizada    │
└──────────────────────┘
```

## Instalación y Configuración

### 1. Clonar el repositorio

```bash
git clone https://github.com/LIZVALMU/Taller_RAGs.git
cd Taller_RAGs
```

### 2. Crear y activar entorno virtual

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual
# En Windows PowerShell:
.\.venv\Scripts\activate

# En Linux/Mac:
source .venv/bin/activate
```

### 3. Instalar dependencias

```bash
# Actualizar pip
python -m pip install --upgrade pip

# Instalar dependencias del proyecto
pip install -q langchain langchain-text-splitters langchain-community bs4
pip install -qU langchain-openai
pip install -qU langchain-pinecone
pip install -q python-dotenv

# Verificar instalación
pip list | grep -E "langchain|openai|pinecone"
```

**Alternativa:** Instalar desde archivo de requisitos
```bash
pip install -r requirements.txt
```

### 4. Configurar variables de entorno

Crea un archivo `.env` en la raíz del proyecto con tus credenciales:

```env
# ================================
# OpenAI Configuration
# ================================
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxx

# ================================
# Pinecone Configuration
# ================================
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxxxxxxxxxx
PINECONE_INDEX_NAME=arep-taller

# ================================
# LangSmith Configuration (Opcional)
# Para debugging y tracing de agentes
# ================================
LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxxxxxxx
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-system-alison-valderrama
```

**Seguridad:** Asegúrate de agregar `.env` a tu `.gitignore` para no exponer tus credenciales.

#### Obtener las API Keys:

| Servicio | URL | Notas |
|----------|-----|-------|
| **OpenAI** | https://platform.openai.com/api-keys | Requiere cuenta con créditos activos |
| **Pinecone** | https://app.pinecone.io/ | Plan gratuito disponible (1 índice serverless) |
| **LangChain** | https://smith.langchain.com/ | Opcional - Para debugging y monitoreo |

### 5. Configurar índice en Pinecone

**Pasos para crear el índice:**

1. Accede a [Pinecone Console](https://app.pinecone.io/)
2. Crea un nuevo índice con la siguiente configuración:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| **Nombre** | `arep` | Debe coincidir con `PINECONE_INDEX_NAME` en `.env` |
| **Dimensión** | `1024` | Compatible con text-embedding-3-large |
| **Métrica** | `cosine` | Métrica de similitud coseno para embeddings |
| **Tipo** | `Serverless` | Sin infraestructura que gestionar |
| **Región** | `AWS us-east-1` | Mejor latencia para la mayoría de usuarios |

3. Espera a que el índice esté en estado `Ready`
4. Copia la API Key desde el dashboard de Pinecone

## Uso del Sistema

### Ejecutar el Notebook

**Opción 1: VS Code (Recomendado)**
1. Abre VS Code en el directorio del proyecto
2. Instala la extensión "Jupyter" de Microsoft
3. Abre el archivo `Taller_Rag_Agent_Lang.ipynb`
4. Selecciona el kernel de Python (`.venv`)
5. Ejecuta las celdas secuencialmente con `Shift + Enter`

**Opción 2: Jupyter Notebook**
```bash
# Instalar Jupyter si no lo tienes
pip install jupyter

# Iniciar Jupyter Notebook
jupyter notebook

# Abre Taller_Rag_Agent_Lang.ipynb desde la interfaz web
```

**Opción 3: JupyterLab**
```bash
# Instalar JupyterLab
pip install jupyterlab

# Iniciar JupyterLab
jupyter lab
```

### Estructura del Notebook

El notebook está organizado en 5 módulos principales:

#### Setup - Configuración del Entorno
- **Instalación de dependencias** necesarias para el sistema RAG
- **Carga de variables de entorno** desde archivo `.env`
- **Inicialización de modelos de IA**:
  - GPT-4o para generación de respuestas
  - text-embedding-3-small para vectorización (512 dims)
- **Configuración de Pinecone** y creación del vector store

#### 1. Pipeline de Indexación
- **1.1 Extracción de Contenido Web**: 
  - WebBaseLoader extrae contenido del blog de Lilian Weng
  - BeautifulSoup filtra elementos relevantes (título, headers, contenido)
- **1.2 Inspección del Contenido**: 
  - Vista previa del documento extraído
- **1.3 Fragmentación Inteligente**: 
  - RecursiveCharacterTextSplitter divide el documento
  - Chunks de 1000 caracteres con overlap de 200
- **1.4 Indexación Vectorial**: 
  - Conversión a embeddings de 512 dimensiones
  - Almacenamiento en Pinecone con metadatos

#### 2. Sistema de Recuperación y Generación
- **2.1 Herramienta de Búsqueda Semántica**: 
  - Función `fetch_relevant_context` para recuperación
  - Top-k=2 documentos más relevantes por consulta
- **2.2 Configuración del Agente RAG**: 
  - Integración de GPT-4o con herramientas de búsqueda
  - System prompt para uso efectivo del contexto

#### 3-4. Demostración y Validación
- **Consulta compleja** con múltiples búsquedas iterativas
- **Consulta simple** para verificación básica
- **Streaming en tiempo real** de respuestas

#### 5. Auditoría del Vector Store
- Estadísticas detalladas del índice Pinecone
- Prueba de búsqueda semántica
- Verificación de salud del sistema

## Ejemplo de Uso

```python
# Realizar una consulta al sistema RAG
user_question = "What is task decomposition?"

print(f"Consulta: {user_question}\n")

for stream_event in rag_agent.stream(
    {"messages": [{"role": "user", "content": user_question}]},
    stream_mode="values",
):
    stream_event["messages"][-1].pretty_print()
```

### Flujo de Procesamiento

```
Usuario hace pregunta
         ↓
┌────────────────────────┐
│   RAG Agent (GPT-4o)   │
│  Analiza la consulta   │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│ fetch_relevant_context │
│ Búsqueda semántica en  │
│   Pinecone (top-k=2)   │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│ Documentos relevantes  │
│   con metadatos y      │
│   similarity scores    │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│   GPT-4o genera        │
│  respuesta basada en   │
│  contexto recuperado   │
└────────┬───────────────┘
         ↓
┌────────────────────────┐
│ Respuesta streaming    │
│ en tiempo real al user │
└────────────────────────┘
```

## Configuración Avanzada

### Ajustar Tamaño de Fragmentos

Modifica el tamaño de los chunks según la naturaleza de tus documentos:

```python
document_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Aumentar para documentos técnicos densos
    chunk_overlap=300,    # Mayor superposición para mejor contexto
    add_start_index=True,
)
```

**Recomendaciones:**
- **Documentos técnicos**: chunk_size=1500-2000, overlap=300-400
- **Artículos cortos**: chunk_size=800-1000, overlap=150-200
- **Documentación legal**: chunk_size=2000-3000, overlap=400-500

### Optimizar Recuperación de Documentos

Ajusta el número de documentos recuperados (k) según precisión vs contexto:

```python
@tool(response_format="content_and_artifact")
def fetch_relevant_context(user_query: str):
    """Recupera información contextual relevante."""
    relevant_docs = rag_vector_store.similarity_search(
        user_query, 
        k=5  # Aumentar para más contexto (pero más tokens)
    )
    # ... resto del código
```

**Trade-offs:**
- **k=2**: Respuestas más precisas, menos contexto
- **k=5**: Más contexto, mayor costo en tokens
- **k=10**: Contexto exhaustivo, riesgo de información irrelevante

### Cambiar Modelo de Lenguaje

Experimenta con diferentes modelos según tus necesidades:

```python
# Opciones disponibles
llm_model = init_chat_model("gpt-4o", model_provider="openai")        # Mejor calidad
llm_model = init_chat_model("gpt-4-turbo", model_provider="openai")   # Balance
llm_model = init_chat_model("gpt-3.5-turbo", model_provider="openai") # Más económico
```

**Comparación de modelos:**

| Modelo | Costo | Calidad | Velocidad | Uso Recomendado |
|--------|-------|---------|-----------|-----------------|
| GPT-4o | Alto | Excelente | Rápido | Producción, análisis complejo |
| GPT-4-turbo | Medio-Alto | Muy buena | Medio | Balance general |
| GPT-3.5-turbo | Bajo | Buena | Muy rápido | Desarrollo, pruebas |

### Configurar Embeddings

Ajusta las dimensiones de los embeddings según tus necesidades:

```python
# 512 dimensiones (recomendado para este proyecto)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    dimensions=512
)

# 1536 dimensiones (mayor precisión, más costoso)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large", 
    dimensions=1536
)
```

**Importante:** Si cambias las dimensiones, debes recrear el índice de Pinecone con la nueva configuración.

## Características Principales del Sistema

### Capacidades Funcionales

-  **Búsqueda Semántica Avanzada**: Encuentra información basándose en significado contextual, no solo coincidencia de palabras clave
-  **Streaming en Tiempo Real**: Respuestas generadas y mostradas progresivamente para mejor UX
-  **Contexto Conversacional**: Mantiene coherencia entre múltiples interacciones consecutivas
-  **Trazabilidad de Fuentes**: Cada respuesta incluye referencias a los documentos fuente utilizados
-  **Búsquedas Iterativas**: El agente puede realizar múltiples búsquedas para consultas complejas
### Características Técnicas

-  **Escalabilidad Horizontal**: Pinecone soporta millones de vectores sin degradación de rendimiento
-  **Arquitectura Modular**: Fácil extensión con nuevas herramientas, fuentes de datos y modelos
-  **Gestión de Embeddings**: Sistema eficiente de vectorización con 512 dimensiones
-  **Optimización de Costos**: Configuración ajustable para balancear calidad vs costo
-  **Serverless**: Sin infraestructura que gestionar, escalado automático

### Seguridad y Confiabilidad

-  **Gestión Segura de Credenciales**: Variables de entorno con python-dotenv
-  **Validación de Inputs**: Verificación de API keys y configuraciones requeridas
-  **Manejo de Errores**: Sistema robusto de logging y error handling
-  **Reducción de Alucinaciones**: Respuestas ancladas en documentos verificables

## Notas Importantes y Consideraciones

### Sobre Pinecone

| Parámetro | Valor Requerido | Notas |
|-----------|----------------|-------|
| **Dimensión** | `1024` | Debe coincidir con text-embedding-3-large |
| **Métrica** | `cosine` | Mejor para embeddings de texto |
| **Región** | `us-east-1` (AWS) | Latencia óptima para América |
| **Tipo** | `Serverless` | Sin gestión de infraestructura |

**Importante**: Si cambias el modelo de embeddings, debes recrear el índice con las nuevas dimensiones.

### Estructura de Costos

#### OpenAI (Precios aproximados - Verificar precios actuales)

| Servicio | Entrada | Salida | Notas |
|----------|---------|--------|-------|
| **GPT-4o** | ~$5.00 / 1M tokens | ~$15.00 / 1M tokens | Última generación |
| **GPT-4-turbo** | ~$10.00 / 1M tokens | ~$30.00 / 1M tokens | Balance calidad/precio |
| **GPT-3.5-turbo** | ~$0.50 / 1M tokens | ~$1.50 / 1M tokens | Opción económica |
| **text-embedding-3-small** | ~$0.02 / 1M tokens | - | Vectorización eficiente |
| **text-embedding-3-large** | ~$0.13 / 1M tokens | - | Mayor precisión |

**Estimación de costos para este proyecto:**
- **Indexación inicial** (~42K tokens): ~$0.001
- **Consulta promedio** (~2K tokens): ~$0.01 - $0.03
- **100 consultas/día**: ~$1 - $3/día

#### Pinecone

- **Plan Gratuito**: 1 índice serverless, suficiente para desarrollo y pruebas
- **Plan Starter**: A partir de $70/mes para producción
- **Consultas**: Incluidas en el plan mensual

### Límites Técnicos y Consideraciones

| Componente | Límite | Recomendación |
|------------|--------|---------------|
| **text-embedding-3-small** | 8,191 tokens/input | Chunks < 1,500 caracteres |
| **GPT-4o** | 128K tokens contexto | Óptimo con k=2-5 documentos |
| **Pinecone Free Tier** | 100K vectores | Suficiente para ~200 artículos |
| **Rate Limits OpenAI** | Varía por tier | Implementar retry logic |

### Consideraciones de Producción

1. **Caché de Embeddings**: Implementar caché para vectores ya generados
2. **Batch Processing**: Procesar múltiples documentos en paralelo
3. **Monitoring**: Usar LangSmith para tracking y debugging
4. **Error Handling**: Implementar reintentos con backoff exponencial
5. **Rate Limiting**: Respetar límites de las APIs

## Solución de Problemas Comunes

### Error: "Invalid API Key" / Autenticación Fallida

**Síntomas:**
```
openai.error.AuthenticationError: Invalid API key provided
```

**Soluciones:**
1. Verifica que la API key de OpenAI sea válida y esté activa
2. Confirma que tu cuenta tenga créditos disponibles en [OpenAI Billing](https://platform.openai.com/account/billing)
3. Verifica que no haya espacios adicionales en el archivo `.env`
4. Recarga las variables de entorno:
   ```python
   from dotenv import load_dotenv
   load_dotenv(override=True)
   ```
5. Reinicia el kernel del notebook

### Error: "Index not found" / Error de Pinecone

**Síntomas:**
```
pinecone.exceptions.NotFoundException: Index 'arep-taller' not found
```

**Soluciones:**
1. Verifica que el nombre del índice en `.env` coincida exactamente con el de Pinecone
2. Confirma que el índice esté en estado "Ready" en [Pinecone Console](https://app.pinecone.io/)
3. Verifica que estés usando la API key correcta del proyecto
4. Espera unos minutos si acabas de crear el índice

### Error: "Dimension mismatch" / Dimensiones Incompatibles

**Síntomas:**
```
ValueError: Dimension mismatch: index has 1536 dimensions, but embeddings have 512
```

**Soluciones:**
1. El índice debe tener exactamente **512 dimensiones** para text-embedding-3-small
2. Si cambiaste el modelo de embeddings, debes recrear el índice:
   ```python
   # Eliminar índice antiguo (⚠️ Cuidado: borra todos los datos)
   pc.delete_index(index_name)
   
   # Crear nuevo índice con dimensiones correctas
   pc.create_index(
       name=index_name,
       dimension=512,
       metric='cosine',
       spec=ServerlessSpec(cloud='aws', region='us-east-1')
   )
   ```

### 🔌 Error: "Connection timeout" / Problemas de Red

**Síntomas:**
```
requests.exceptions.ConnectionError: Connection timeout
```

**Soluciones:**
1. Verifica tu conexión a internet
2. Comprueba que no haya firewall bloqueando las APIs
3. Implementa reintentos con backoff:
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   def make_api_call():
       # Tu código aquí
       pass
   ```

### Error: "Rate limit exceeded" / Límite de Tasa Excedido

**Síntomas:**
```
openai.error.RateLimitError: Rate limit exceeded
```

**Soluciones:**
1. Reduce la frecuencia de llamadas a la API
2. Implementa un sistema de cola con delays
3. Considera actualizar tu tier en OpenAI para mayores límites
4. Usa batch processing para procesar múltiples documentos eficientemente

### Otros Problemas Comunes

| Problema | Solución |
|----------|----------|
| **Módulo no encontrado** | `pip install -r requirements.txt` |
| **Kernel crash** | Reinicia el kernel y ejecuta celdas secuencialmente |
| **Respuestas vacías** | Verifica que el índice tenga documentos: `index.describe_index_stats()` |
| **Costos altos** | Reduce `k` en similarity_search o usa GPT-3.5-turbo |

## Recursos y Referencias

### Documentación Oficial

| Recurso | URL | Descripción |
|---------|-----|-------------|
| **LangChain Docs** | https://python.langchain.com/ | Framework principal del proyecto |
| **OpenAI Platform** | https://platform.openai.com/docs/ | API reference completa |
| **Pinecone Docs** | https://docs.pinecone.io/ | Base de datos vectorial |
| **LangSmith** | https://docs.smith.langchain.com/ | Debugging y monitoring |

### Tutoriales y Guías

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) - Tutorial oficial de RAG
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Guía completa de RAG
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) - Guía de embeddings
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/) - Documentación de agentes

### Artículos Relevantes

- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) - Blog de Lilian Weng (fuente de datos del proyecto)
- [RAG vs Fine-tuning](https://www.pinecone.io/learn/rag-vs-finetuning/) - Comparación de técnicas
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database/) - Comparación de bases de datos vectoriales

### Videos Educativos

- [LangChain Crash Course](https://www.youtube.com/watch?v=LbT1yp6quS8) - Intro a LangChain
- [RAG Explained](https://www.youtube.com/watch?v=T-D1OfcDW1M) - Explicación de RAG
- [Pinecone Tutorial](https://www.youtube.com/watch?v=gTCU9I6QqCE) - Tutorial de Pinecone

### Repositorios Relacionados

- [LangChain Examples](https://github.com/langchain-ai/langchain/tree/master/docs/docs/tutorials) - Ejemplos oficiales
- [RAG Cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/vector_databases/pinecone/Gen_QA.ipynb) - OpenAI cookbook
- [Awesome RAG](https://github.com/awesome-rag/awesome-rag) - Lista curada de recursos RAG

## Autora

**Alison Geraldine Valderrama Munar**

**Contacto:** AREP - Arquitecturas Empresariales  
**Institución:** Escuela Colombiana de Ingeniería Julio Garavito  
**Repositorio:** https://github.com/LIZVALMU/Taller_RAGs

---

## Licencia
Este proyecto fue desarrollado como parte del curso de Arquitecturas Empresariales de la Escuela Colombiana de Ingeniería Julio Garavito.

---

<div align="center">

**Si este proyecto te fue útil, no olvides darle una estrella en GitHub**

Desarrollado por Alison Valderrama

</div>  