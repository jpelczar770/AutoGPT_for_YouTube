import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = apikey

st.title("YouTube GPT Creator 🦜️🔗")

prompt = st.text_input("Plug in your prompt here")

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'

)
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'write me a youtube video script based on this title TITLE: {title}'
)

memory = ConversationBufferMemory(input_key = 'topic', memory_key='chat history')
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=memory)
sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

#

if prompt:
    response = sequential_chain({'topic': prompt})
    st.write(response['title'])
    st.write(response['script'])
    with st.expander('Chat History'):
        st.info(memory.buffer)


        #testing
