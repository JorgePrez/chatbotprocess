#################################

import boto3
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_aws import ChatBedrock, AmazonKnowledgeBasesRetriever
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from operator import itemgetter
from langchain_core.prompts import MessagesPlaceholder
import botocore.exceptions  # Importamos excepciones específicas de Boto3
from typing import List, Dict
from pydantic import BaseModel  ##Importante esto a veces no es compatible
from langchain.schema import Document
from langchain.schema.runnable import RunnableLambda

#from langchain_openai.chat_models import ChatOpenAI

from collections import defaultdict
from langchain.schema import Document  # Asegúrate de importar Document si es necesario
from dotenv import load_dotenv
import os


class Citation(BaseModel):
    page_content: str
    metadata: Dict

def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.page_content, metadata=doc.metadata ) for doc in response]

# ------------------------------------------------------
# S3 Presigned URL, esto permite realizar descargar del documento

def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    """Generate a presigned URL to share an S3 object"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except NoCredentialsError:
        st.error("AWS credentials not available")
        return ""
    return response

def parse_s3_uri(uri: str) -> tuple:
    """Parse S3 URI to extract bucket and key"""
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key


# Configuración de Bedrock
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-haiku-20240307-v1:0"
model_kwargs = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

# Generador de respuesta ChatBedrock:
llmClaude = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

# Cargar variables del archivo .env
load_dotenv()




SYSTEM_PROMPT = (
      "Eres un asistente especializado en explicar los **procesos de la Universidad Francisco Marroquín (UFM)**. "
            "Eres amigable y servicial. Responde a los saludos de manera cordial. Responde las consultas sobre procesos "
            "basándote únicamente en el siguiente contexto:\n{context}\n"
            "**Obligatorio:** No busques en ninguna otra fuente de información ni uses tu propio conocimiento para formular ideas.\n\n"

            "### Consulta de usuario ###\n"
            "Siempre que recibas una consulta, debes hacer una pregunta de aclaración solicitando más contexto y proporcionando "
            "una lista de los posibles procesos relacionados con la consulta, ordenados por prioridad de mayor a menor. La estructura "
            "de la respuesta inicial será la siguiente:\n\n"

            "1. **Procesos relacionados:**\n"
            " Haz una pregunta para pedir más detalles o confirmar el proceso específico que el usuario desea obtener. Por ejemplo:\n"
            " 'Para poder ayudarte mejor, ¿podrías confirmar cuál de los siguientes procesos es el que te interesa?'\n\n"
            " *Lista de procesos relacionados: Muestra una listado con los procesos más relevantes, ordenados por prioridad. Usa el siguiente formato:*\n\n"
            "   **Ejemplo:**\n"
            "   - UFM-ADM-004 - Proceso A\n"
            "   - UFM-IT-020 - Proceso B \n"
            "   - UFM-REG-038 - Proceso C \n\n"
            "   - UFM-PU-010 - Proceso D \n\n"
            "   - UFM-CTAC-006 - Proceso E \n\n"

            "2. **Confirmación:** (Mostrar un mensaje al usuario para que eliga uno de los procesos mostrados, solicitarle que escriba el codigo o nombre del proceso, además mencionar que si el proceso que busca no se encuentra puede ampliarse el listado de procesos relacionados) \n"
            "   - Una vez que el usuario confirme el proceso que le interesa, procede a entregar la información detallada siguiendo "
            "los pasos descritos en la sección 'Pasos Obligatorios'.\n\n"

            "### Pasos Obligatorios (Una vez confirmada la selección) ###\n"
            "Cuando el usuario confirme el proceso que desea conocer, sigue estrictamente los siguientes pasos (formatea la respuesta de manera clara y organizada):\n\n"
            
            "1. **Identificación del proceso:** Coloca el nombre y el código del proceso en **negrita** usando markdown.\n\n"
            "2. **Enlace a flujograma:** muestra el enlace clickeable con el texto 'Ver flujograma' ( *Obligatorio utilizar la url del campo link-flujograma, NUNCA MUESTRES LA IMAGEN solo deseo el link clickeable*)  \n\n"
            "3. **Enlace a documento de pasos:** muestra el enlace clickeable con el texto 'Ver documento de pasos' ( *Obligatorio utilizar la url del campo link-documento-pasos* ) \n\n "
            "4. **Objetivo:**  Descripción del objetivo o propósito del proceso. \n\n"

            "5. **Explicación detallada**\n"
            "    **Pasos del proceso:** Asegúrate de identificar **todos los pasos** desde el inicio hasta el final del proceso (por ejemplo, del paso 1 al paso 11).\n"
            "    **Instrucciones para asegurar que no se omita ningún paso:** \n" 
            "     - **Verifica que todos los pasos consecutivos estén presentes**, desde el primero hasta el último (por ejemplo, si el proceso tiene 11 pasos, asegúrate de que todos los pasos del 1 al 11 estén incluidos). \n" 
            "     - **Recorre todo el contenido disponible** y **combina la información fragmentada** si un paso está dividido en varias partes o aparece en diferentes secciones. \n" 
            "     - **No detengas la búsqueda hasta que identifiques todos los pasos declarados en la numeración completa.** Si los pasos están desordenados o fragmentados, organiza y presenta los pasos de manera secuencial (1, 2, 3,... 11). \n"  

            "      **Presenta cada paso siguiendo el formato a continuación:** \n"
            "     - 1. [Nombre del paso]:\n"
            "         - **Descripción:** Explica el paso en detalle.\n"
            "         - **Tiempos:** Indica el tiempo estimado para completar el paso.\n"
            "         - **No negociables:** Cosas que no se pueden omitir o cambiar.\n"
            "         - **Participantes:** Personas o áreas involucradas.\n"
            "     - 2. [Nombre del paso]:\n"
            "         - **Descripción:** Explica el paso en detalle.\n"
            "         - **Tiempos:** Indica el tiempo estimado para completar el paso.\n"
            "         - **No negociables:** Cosas que no se pueden omitir o cambiar.\n"
            "         - **Participantes:** Personas o áreas involucradas.\n\n"

            "     - **Repite este formato para cada paso**. Asegúrate de incluir todos los pasos relacionados con el proceso y de que cada subelemento (como tiempos, no negociables, participantes) "
            "esté separado con saltos de línea para mantener la estructura clara y organizada.\n\n"

            "### Listado Completo por Unidad ###\n"
            "Si el usuario solicita ver un listado completo de los procesos de una unidad, responde con el siguiente formato:\n\n"
            
            "1. **Listado Completo de Procesos:**\n"
            "   - Presenta todos los procesos disponibles en la unidad solicitada.\n"
            "   - Utiliza el siguiente formato para cada proceso: (Repite este formato para cada proceso): \n"
            "       1. Nombre del proceso (Código del proceso)\n"
            "      \n"
            
            "   **Ejemplo:**\n"
            "   1. UFM-IT-001 - Proceso A\n"
            "   2. UFM-IT-002 - Proceso B \n"
            "   3. UFM-IT-003 - Proceso C \n\n"

            "   **Nota:** Si no se encuentran procesos para la unidad solicitada, responde: 'No se encontraron procesos para la unidad\n\n"

       
            "### Manejo de Consultas sin Información Relevante ###\n"
            "- Si no hay procesos disponibles en el contexto que coincidan con la consulta, responde de manera clara:\n"
            "  'Lo siento, no se encontró información relevante para tu consulta en el contexto proporcionado.'\n\n"

            "### Manejo de Respuestas Cortas ###\n"
            "- Si la consulta solo requiere un enlace o un dato específico (nombre o código de proceso), proporciona únicamente esa información sin desglosar todos los pasos.\n\n"

)


##Imporante el retriever 
##Unidades a las que se tiene acceso

#"in": { "key": "animal", "value": ["cat", "dog"] }

# Si no hay ninguno seleccionado.
##  "value": ["IT", "AES"]

#In, este filtro no esta activo
# notIn
# Arreglo con áreas y su estado activo/inactivo, esto es un prototipo de permisos
#Lo ideal es que se consultar a la base de datos 


areas_codigos = [
    {"codigo": "ADM", "nombre": "Admisiones", "activo": True},
    {"codigo": "AES", "nombre": "Atención al estudiante", "activo": True},
    {"codigo": "CETA", "nombre": "CETA", "activo": True},
    {"codigo": "CSM", "nombre": "Clínicas Salud", "activo": True},
    {"codigo": "COLAB", "nombre": "CoLab", "activo": True},
    {"codigo": "CON", "nombre": "Contabilidad", "activo": True},
    {"codigo": "CE", "nombre": "Crédito Educativo", "activo": True},
    {"codigo": "CTAC", "nombre": "Cuenta Corriente", "activo": True},
    {"codigo": "IT", "nombre": "Tecnología", "activo": True},
    {"codigo": "MERC", "nombre": "Mercadeo", "activo": True},
    {"codigo": "PU", "nombre": "Publicaciones", "activo": True},
    {"codigo": "REG", "nombre": "Registro", "activo": True},
    {"codigo": "RH", "nombre": "Recursos Humanos", "activo": True},
    {"codigo": "TES", "nombre": "Tesorería", "activo": True},
    {"codigo": "UFM-LABS", "nombre": "UFM LABS", "activo": True},
]

# Construimos el mensaje con el listado de áreas activas
mensaje_areas = "\nÁreas disponibles:\n"
for area in areas_codigos:
    if area["activo"]:
        mensaje_areas += f"- **{area['nombre']}** ({area['codigo']})\n"


# Función que genera la configuración completa del retriever
def generar_configuracion_retriever():
    # Obtener los códigos de áreas inactivas
    inactivas = [area["codigo"] for area in areas_codigos if not area["activo"]]

    # Construir la configuración base
    config = {
        "vectorSearchConfiguration": {
            "numberOfResults":  100  #maximo 100
        }
    }

    # Si hay áreas inactivas, agregar el filtro
    if inactivas:
        config["vectorSearchConfiguration"]["filter"] = {
            "notIn": {
                "key": "codigo_area",
                "value": inactivas
            }
        }

    return config

# Crear el retriever con la configuración generada
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="I9HQYMMI4A",
    retrieval_config=generar_configuracion_retriever()
)


# Función para crear el prompt dinámico
def create_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]
    )

#################################################################################################

# Función para recuperar y depurar el contexto

def obtener_contexto(inputs):
    question = inputs["question"]  # Extraer la pregunta
    documentos = retriever.invoke(question)  # Obtener documentos relevantes

    # Diccionario para agrupar contenido por 'identificador_proceso'
    procesos_agrupados = defaultdict(list)

    # Iterar sobre los documentos recuperados
    for doc in documentos:
        # Extraer los metadatos y el identificador
        source_metadata = doc.metadata.get('source_metadata', {})
        identificador = source_metadata.get('identificador_proceso', 'Sin ID')

        # Agrupar el contenido del documento bajo el mismo identificador
        procesos_agrupados[identificador].append(doc.page_content)

    # Crear una nueva lista de Documentos con contenido concatenado por identificador_proceso
    documentos_concatenados = []
    for identificador, contenidos in procesos_agrupados.items():
        # Concatenar los contenidos
        contenido_concatenado = "\n".join(contenidos)

        # Tomar los metadatos y el score del primer documento con este identificador
        doc_base = next(
            (doc for doc in documentos if doc.metadata.get('source_metadata', {}).get('identificador_proceso') == identificador),
            None
        )

        if doc_base:
            metadatos_base = doc_base.metadata  # Metadatos del primer documento
            score_base = doc_base.metadata.get('score', 0)  # Score del primer documento
        else:
            metadatos_base = {}
            score_base = 0

        # Incluir el score en los metadatos base
        metadatos_base['score'] = score_base

        # Crear un nuevo objeto Document con el contenido concatenado
        documento_concatenado = Document(
            metadata=metadatos_base,
            page_content=contenido_concatenado
        )

        # Agregar el documento a la lista final
        documentos_concatenados.append(documento_concatenado)

    # Imprimir para verificar el resultado
    #for doc in documentos_concatenados:
    #    print(f"Documento Concatenado (ID: {doc.metadata['source_metadata'].get('identificador_proceso', 'Sin ID')}):")
    #    print(f"Metadata: {doc.metadata}")
    #    print(f"Score: {doc.metadata['score']}")
    #    print(f"Contenido (primeros 500 caracteres): {doc.page_content[:500]}...\n")
    
    #print(documentos_concatenados)

    return documentos_concatenados  # Devolver los documentos concatenados



# Crear el pipeline con depuración
context_pipeline = RunnableLambda(obtener_contexto)


# Chain con historial de mensajes
def create_chain_with_history():
    prompt = create_prompt_template()

    chain = (
        RunnableParallel({
            "context": context_pipeline,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        })
        .assign(response=prompt | llmClaude   | StrOutputParser())
        .pick(["response", "context"])
    )

    #print(chain)

    history = StreamlitChatMessageHistory(key="chat_messages")
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )


# Chain con historial de mensajes
def create_chain_with_history_old():
    prompt = create_prompt_template()

    chain = (
        RunnableParallel({
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        })
        .assign(response=prompt | llmClaude   | StrOutputParser())
        .pick(["response", "context"])
    )

    #print(chain)

    history = StreamlitChatMessageHistory(key="chat_messages")
    return RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )

# Función para limpiar historial de chat
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": f"Soy tu asistente sobre procesos de la UFM .\n{mensaje_areas} \n"}]




# Función para manejar errores de Bedrock
def handle_error(error):
    st.error("Ocurrió un problema. Por favor, repite tu consulta.")
  #  st.write("Detalles técnicos (para depuración):")
  #  st.code(str(error))  # Mostrar los detalles del error para propósitos de depuración

# Configurar la interfaz de Streamlit
def main():
    st.set_page_config(page_title='Procesos UFM 🔗'
 )

    if "messages" not in st.session_state:
        clear_chat_history()

    streaming_on = True ##st.sidebar.checkbox('Streaming (Mostrar generación de texto en tiempo real)', value=True)
    st.sidebar.button('Limpiar pantalla', on_click=clear_chat_history)


        # Mostrar historial de mensajes en la barra lateral
 

    st.divider()
    #st.write("History Logs(Debug)")
    #st.sidebar.header("Historial de Mensajes")
    #for i, msg in enumerate(st.session_state.messages):
    #    st.sidebar.write(f"{i + 1}. **{msg['role'].capitalize()}**: {msg['content']}")
   

    chain_with_history = create_chain_with_history()

    #print(chain_with_history)


    #dynamic_prompt = create_prompt_template().format(history=st.session_state.messages, question=SYSTEM_PROMPT)
    #st.write("**Prompt utilizado:**")
    #st.code(dynamic_prompt)

    #st.write(SYSTEM_PROMPT)
    #st.code(SYSTEM_PROMPT)



    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        config = {"configurable": {"session_id": "any"}}

        try:
            if streaming_on:
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ''
                    for chunk in chain_with_history.stream(
                        {"question": prompt, "history": st.session_state.messages},
                        config
                    ):
                        if 'response' in chunk:
                            full_response += chunk['response']
                            placeholder.markdown(full_response)
                        else:
                            full_context = chunk['context']
                    placeholder.markdown(full_response)

                    citations = extract_citations(full_context)
                    #st.write(full_context)

                    #with st.expander("Mostrar fuentes >"):
                    # for citation in citations:
                    #    st.write("**Contenido:** ", citation.page_content)
                    #    s3_uri = citation.metadata['location']['s3Location']['uri']
                    #    bucket, key = parse_s3_uri(s3_uri)
                    #    presigned_url = create_presigned_url(bucket, key)
                    #    if presigned_url:
                    #            st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                    #    else:
                    #        st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    #    st.write("**Score**:", citation.metadata['score'])
                    #    st.write("--------------")


                    
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
        except botocore.exceptions.BotoCoreError as e:
            handle_error(e)  # Manejamos los errores de Boto3 aquí
        except Exception as e:
            handle_error(e)  # Capturamos cualquier otro error inesperado

if __name__ == "__main__":
    main()