import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ['OPENAI_API_KEY'] = apikey

st.title("YouTube GPT Creator 🦜️🔗")

prompt = st.text_input("Plug in your prompt here")

title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube video title about {topic}'

)
script_template = PromptTemplate(
    input_variables = ['title'],
    template = 'write me a youtube video script based on this title TITLE: {title}'
)

llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)


if prompt:
    response = sequential_chain.run(topic=prompt)
    st.write(response)
