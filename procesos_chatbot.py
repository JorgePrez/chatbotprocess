# ------------------------------------------------------
# Streamlit
# Knowledge Bases con Amazon Bedrock y LangChain 🦜️🔗
# ------------------------------------------------------

import boto3

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import BedrockChat
from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
import streamlit as st

import logging

from typing import List, Dict
from pydantic import BaseModel
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory


# ------------------------------------------------------
# Amazon Bedrock - settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

#model_id = "anthropic.claude-instant-v1"


model_kwargs =  { 
        "max_tokens": 2048,
        "temperature": 0.0,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman"],
    }

logging.getLogger().setLevel(logging.ERROR) # reduce log level



class Citation(BaseModel):
    page_content: str
    metadata: Dict

def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.page_content, metadata=doc.metadata) for doc in response]

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


########################################################################################################################################################################

def todos():

    # ------------------------------------------------------
    # LangChain - RAG chain with chat history


    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system", 
            (
                "Eres un asistente especializado en explicar los **procesos de IT de la Universidad Francisco Marroquín (UFM)**. Eres amigable y servicial "
                "Responde a preguntas como 'hola', '¿cómo estás?', ' que tal' , 'buenos días', etc., de manera cordial, ya que usualmente los usuarios comienzan saludando "
                "Responde las consultas sobre procesos basándote únicamente en el siguiente contexto:\n{context} "
               "\nCuando identifiques un proceso que tenga relación con la consulta, **debes seguir estrictamente los siguientes pasos** (formatea la respuesta de manera clara y organizada):\n"
                "1. **Identifica el nombre y código del proceso mencionado** (colócalo en **negrita** para destacarlo).\n"
                "2. **Proporciona una explicación detallada** que incluya lo siguiente:\n"
                "   - El **objetivo** del proceso.\n"
                "   - Los **participantes clave** involucrados.\n"
                "   - Las **etapas del proceso**, con descripciones claras de cada una.\n"
                "   - Los **tiempos estimados** de cada etapa o de todo el proceso.\n"
                "   - Las **decisiones clave** que se deben tomar en el proceso.\n"
                "   - **Consideraciones adicionales** que puedan ser relevantes, si aplican.\n"
                "3. **Explica el flujograma** del proceso, destacando los puntos importantes y cómo estos se relacionan con las fases del proceso descritas. \n"
                "4.  Si en la metadata existe el campo 'link-flujograma', muestra un enlace clickeable con el texto 'Ver flujograma' **NO DEBES INVENTAR UNA URL SIEMPRE USA LA DE LA METADATA**. Si el campo 'link-flujograma' no está presente en la metadata, omite el enlace y no lo muestres\n\n"
                "**Importante**: Si no puedes identificar un proceso claro y relevante, responde siempre indicando que no puedes proporcionar información sobre ese tema."
                "**Importante**: No respondas preguntas que no estén relacionadas con los procesos de la Universidad Francisco Marroquín UFM. Si la consulta está fuera de este dominio, indica que no puedes proporcionar información sobre ese tema."
            )
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ]
)

    

  
 




# Mostrar el contenido del prompt para verlo
    #print(prompt)
    
    

    # PAQSWKEITI - PDF PERO SIN PNG
    # 2LJ8Q2CO0H - FULL WORD
    # PBXHV68Y30 - FULL PDF
    # ZICV33XA4X  - FULL PDF 2  Ya no exi
    #NXHVZAH2VO - TODOS DE IT



    # Amazon Bedrock - KnowledgeBase Retriever 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id= "NXHVZAH2VO", #"PAQSWKEITI", #  Knowledge base ID
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 50}}, #aqui siempre deberia salir el que coincide con el nombre 
        #ojo que pasa si piden la explicación de más de 1 proceso
    )
    ##Imporante, para que devuelva todos ese número debería ser más grande, en el caso que quisiera un listado por ejemplo.
    ## No es necesario mostrar estas referenciaas únicamente sacar el link de allí y mostrarlo en pantalla

    model = ChatBedrock(
        client=bedrock_runtime,
        model_id=model_id,
        model_kwargs=model_kwargs,
    )

    chain = (
        RunnableParallel({
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
            "history": itemgetter("history"),
        })
        .assign(response = prompt | model | StrOutputParser())
        .pick(["response", "context"])
    )

    # Streamlit Chat Message History
    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chain with History
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )


    import streamlit as st


# Clear Chat History function
    def clear_chat_history():
        history.clear()
        st.session_state.messages = [{"role": "assistant", "content": "Soy tu asistente de procesos IT de la UFM"}]

    with st.sidebar:
       # st.title('Asistente de procesos')
        streaming_on = st.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
        st.button('Limpiar pantalla', on_click=clear_chat_history)
        #st.divider()
        #st.write("History Logs(Debug)")
        #st.write(history.messages)

    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Soy tu asistente de procesos IT de la UFM"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat Input - User Prompt 
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        config = {"configurable": {"session_id": "any"}}
        
        if streaming_on:
            # Chain - Stream
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ''
                for chunk in chain_with_history.stream(
                    {"question" : prompt, "history" : history},
                    config
                ):
                    if 'response' in chunk:
                        full_response += chunk['response']
                        placeholder.markdown(full_response)
                    else:
                        full_context = chunk['context']
                placeholder.markdown(full_response)
                # Citations with S3 pre-signed URL
                citations = extract_citations(full_context)
                

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": full_response})

                ##print (full_context) Esto permite debuggear
        else:
            # Chain - Invoke
            with st.chat_message("assistant"):
                response = chain_with_history.invoke(
                    {"question" : prompt, "history" : history},
                    config
                )
                st.write(response['response'])
                # Citations with S3 pre-signed URL
                citations = extract_citations(response['context'])
               

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": response['response']})

                ##print (full_context)


#PAGES = {
#    "Todos" : todos
#}

def main():
    #import streamlit as st
    # Título de la página
    st.set_page_config(page_title='Procesos UFM-IT 🔗')
   # st.sidedar.title('Navigation')
   # choice = st.sidebar.selectbox("--", list(PAGES.keys()))
   # PAGES[choice]()
    todos()

if __name__ == "__main__":
    main()