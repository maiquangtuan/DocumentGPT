import os

import openai
import gradio as gr

from document_loader.pdf import extract_text_from_pdffile, extract_text_from_pdflink
from utils import split_text
from retriever.hybrid_search import SemanticSearchHybrid



def load_recommender_with_file(pdf_file_path):
    global recommender
    texts = extract_text_from_pdffile(pdf_file_path)
    chunks = split_text(texts)
    recommender = SemanticSearchHybrid(chunks)
    return 'Corpus Loaded.'

def load_recommender_with_link(pdf_link):
    global recommender
    texts = extract_text_from_pdflink(pdf_link)
    chunks = split_text(texts)
    recommender = SemanticSearchHybrid(chunks)
    return 'Corpus Loaded.'


messages=[{"role": "system", "content": "you are a helpful assistant who can read and retrieve information from documents"},]

def chatbot_response(openAI_key, msg):
    item =  {"role": "user", "content": msg}
    messages.append(item)
    openai.api_key = openAI_key
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, temperature = 0.2)
    return str(response['choices'][0]['message']['content'])

def generate_answer(openAI_key, question):
    topn_chunks = recommender(question)
    context = ""
    context  += 'search results:\n\n'
    for c in topn_chunks:
        context += c + '\n\n'
        
    instruction = f"Compose a comprehensive reply to the query using the given search results.\n\n Question: {question} "\
    
    user_msg = context + instruction
    answer = chatbot_response(openAI_key, user_msg)
    return answer


def question_answer(url, file, question,openAI_key):
    if openAI_key.strip()=='':
        return '[ERROR]: Please enter you Open AI Key. Get your key here : https://platform.openai.com/account/api-keys'
    if url.strip() == '' and file == None:
        return '[ERROR]: Both URL and PDF is empty. Provide atleast one.'
    
    if url.strip() != '' and file != None:
        return '[ERROR]: Both URL and PDF is provided. Please provide only one (eiter URL or PDF).'

    if url.strip() != '':
        glob_url = url
        load_recommender_with_link(glob_url)

    else:
        old_file_name = file.name
        file_name = file.name
        file_name = file_name[:-12] + file_name[-4:]
        os.rename(old_file_name, file_name)
        load_recommender_with_file(file_name)

    if question.strip() == '':
        return '[ERROR]: Question field is empty'

    return generate_answer(openAI_key, question)



title = 'DocumentGPT'
description = """DocumentGPT is a tool for chatting with document, inspired from https://github.com/bhaskatripathi/pdfGPT. In this Repo I implement the same idea using Gradio, but the retriever is using Hybrid Search (BM25 + SBERT) instead of universal encoder. The chatbot is using OpenAI GPT-3.5-turbo. Using BM25 only will result in faster response, but the answer will be less relevant."""

with gr.Blocks() as demo:

    gr.Markdown(f'<center><h1>{title}</h1></center>')
    gr.Markdown(description)

    with gr.Row():
        
        with gr.Group():
            gr.Markdown(f'<p style="text-align:center">Get your Open AI API key <a href="https://platform.openai.com/account/api-keys">here</a></p>')
            openAI_key=gr.Textbox(label='Enter your OpenAI API key here')
            url = gr.Textbox(label='Enter PDF URL here')
            gr.Markdown("<center><h4>OR<h4></center>")
            file = gr.File(label='Upload your PDF/ Research Paper / Book here', file_types=['.pdf'])
            question = gr.Textbox(label='Enter your question here')
            btn = gr.Button(value='Submit')
            btn.style(full_width=True)

        with gr.Group():
            answer = gr.Textbox(label='The answer to your question is :')
        btn.click(question_answer, inputs=[url, file, question,openAI_key], outputs=[answer])
demo.launch()

