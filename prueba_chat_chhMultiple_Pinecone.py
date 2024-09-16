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

# Se probará si es suficiente con este modelo de lo contrario se hara la prueba con otros.
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
            ("system", "You are a helpful assistant, always answer in Spanish"
            "Answer the question based only on the following context:\n {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Amazon Bedrock - KnowledgeBase Retriever 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="KZ0SO65RI1", #  Knowledge base ID
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

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

# Page title
   # st.set_page_config(page_title='Chatbot CHH')

# Clear Chat History function
    def clear_chat_history():
        history.clear()
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

    with st.sidebar:
        st.title('Todos los autores 🔗')
        streaming_on = st.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
        st.button('Limpiar pantalla', on_click=clear_chat_history)
        st.divider()
        st.write("History Logs")
        st.write(history.messages)

    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")
                   
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": full_response})
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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": response['response']})


########################################################################################################################################################################


def mises_knowledge():

    # ------------------------------------------------------
    # LangChain - RAG chain with chat history

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, always answer in Spanish"
            "Answer the question based only on the following context:\n {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Amazon Bedrock - KnowledgeBase Retriever 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="4L0WE8NOOH", #  Knowledge base ID
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

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

# Page title
   # st.set_page_config(page_title='Chatbot CHH')

# Clear Chat History function
    def clear_chat_history():
        history.clear()
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

    with st.sidebar:
        st.title('Ludwig von Mises 🔗')
        streaming_on = st.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
        st.button('Limpiar pantalla', on_click=clear_chat_history)
        st.divider()
        st.write("History Logs")
        st.write(history.messages)

    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                    #    presigned_url = create_presigned_url(bucket, key)
                    #    if presigned_url:
                    #        st.markdown(f"Fuente: [{s3_uri}]({presigned_url})")
                    #    else:
                        
                        st.write(f"**Fuente**: *{key}* ")
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": full_response})
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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")

                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": response['response']})



########################################################################################################################################################################


def hayek_knowledge():

    # ------------------------------------------------------
    # LangChain - RAG chain with chat history

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, always answer in Spanish"
            "Answer the question based only on the following context:\n {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Amazon Bedrock - KnowledgeBase Retriever 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="HME7HA8YXX", #  Knowledge base ID
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

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

# Page title
   # st.set_page_config(page_title='Chatbot CHH')

# Clear Chat History function
    def clear_chat_history():
        history.clear()
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

    with st.sidebar:
        st.title('Friedrich A. Hayek 🔗')
        streaming_on = st.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
        st.button('Limpiar pantalla', on_click=clear_chat_history)
        st.divider()
        st.write("History Logs")
        st.write(history.messages)

    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": full_response})
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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": response['response']})


########################################################################################################################################################################


def hazlitt_knowledge():

    # ------------------------------------------------------
    # LangChain - RAG chain with chat history

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant, always answer in Spanish"
            "Answer the question based only on the following context:\n {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )

    # Amazon Bedrock - KnowledgeBase Retriever 
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id="7MFCUWJSJJ", #  Knowledge base ID
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 10}},
    )

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

# Page title
   # st.set_page_config(page_title='Chatbot CHH')

# Clear Chat History function
    def clear_chat_history():
        history.clear()
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

    with st.sidebar:
        st.title('Henry Hazlitt 🔗')
        streaming_on = st.toggle('Streaming (Mostrar generación de texto en tiempo real)',value=True)
        st.button('Limpiar pantalla', on_click=clear_chat_history)
        st.divider()
        st.write("History Logs")
        st.write(history.messages)

    # Initialize session state for messages if not already present
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")
                       
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": full_response})
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
                with st.expander("Mostrar fuentes >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        s3_uri = citation.metadata['location']['s3Location']['uri']
                        bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                        st.write(f"**Fuente**: *{key}* ")
                  
                        st.write("**Score**:", citation.metadata['score'])
                        st.write("--------------")

                # session_state append
                st.session_state.messages.append({"role": "assistant", "content": response['response']})


########################################################################################################################################################################


PAGES = {
    "Todos" : todos,
    "Friedrich A. Hayek" : hayek_knowledge,
    "Ludwig von Mises" : mises_knowledge,
    "Henry Hazlitt" : hazlitt_knowledge,
}

def main():
    #import streamlit as st
    # Título de la página
    st.set_page_config(page_title='Chatbot CHH 🔗')
   # st.sidedar.title('Navigation')
    choice = st.sidebar.selectbox("Generar respuestas según", list(PAGES.keys()))
    PAGES[choice]()

if __name__ == "__main__":
    main()