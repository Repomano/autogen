import concurrent.futures
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import torch
import torch.nn.functional as F
#from pinecone import Pinecone, ServerlessSpec  
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, AutoModel, BitsAndBytesConfig, LlamaForCausalLM, pipeline
# importing the module  
#from pytube import YouTube  
import networkx as nx
import torch.nn as nn
import matplotlib.pyplot as plt
import time
import json
import requests
import threading
import tiktoken as tiktoken
import pinecone
import torch
import os
import numpy as np
import openai
from openai import OpenAI

MAX_THREADS = 5
SAVE_PATH = "/home/matt/Documents/GitHub/coder-agent/scraper/codes" #to_do  
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "...")
PINECONE_API_KEY="..."
objective=f"""Using pytorch, pandas, matplotlib, tensorflow and all ML libraries.
                The objective of the work is to forecast the number of passengers arriving at the various security gates of Fiumicino and Ciampino airports.
                Forecasts must be calculated with:
                • Time depth of one month
                • 30 minute granularity
                \n After having performed some data cleaning and preparation, develop a model to predict the values in dataframe from file located in folder
                '/home/matt/Documents/GitHub/stored-procedure-bot/babyagi/y_T1ovest30_gen2023.csv'.
                \n Use the split of /home/matt/Documents/GitHub/stored-procedure-bot/babyagi/y_T1ovest30_gen2023.csv for training max 10 epoch.
                Note carefully that from 3rd column there number of passengers going through a terminal gate every 30 minutes.
                \n Make forecast of these values.
                Theses are the two tables to be used for the task and represnt the number of person arriving to a terminal every 30 minutes.\n one table for train and the othe rfor validation from y_T1ovest30_gen2023. \n
                The first table contains the following columns:\n
                y_T1ovest30_gen2023 validation set is:\n
                COD_VETT_VOLO,DAY,00:00,00:30,01:00,01:30,02:00,02:30,03:00,03:30,04:00,04:30,05:00,05:30,06:00,06:30,07:00,07:30,08:00,08:30,09:00,09:30,10:00,10:30,11:00,11:30,12:00,12:30,13:00,13:30,14:00,14:30,15:00,15:30,16:00,16:30,17:00,17:30,18:00,18:30,19:00,19:30,20:00,20:30,21:00,21:30,22:00,22:30,23:00,23:30
                \n with values in format:\n
                4Y02053,2023-02-26,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,5.0,73.0,79.0,17.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0
                """
searched_pages=[]
pages = []
visited_pages = []
links = []
device_map = {"": 0}


class DAGNode:
    def __init__(self, value):
        self.value = value
        self.children = set()
        self.parents = set()  # Add a set to store parent nodes

class DAG:
    def __init__(self):
        self.nodes = {}

    def add_node(self, value):
        if value not in self.nodes:
            self.nodes[value] = DAGNode(value)

    def add_edge(self, parent, child):
        self.add_node(parent)
        self.add_node(child)
        self.nodes[parent].children.add(child)

    def __str__(self):
        result = []
        for node in self.nodes.values():
            for child in node.children:
                result.append(f"{node.value} -> {child}")
        return "\n".join(result)
    
    def to_networkx(self):
        G = nx.DiGraph()

        for node in self.nodes.values():
            for child in node.children:
                G.add_edge(node.value, child)

        return G

    def populate_parents(self):
        for parent, node in self.nodes.items():
            #print(parent,node)
            for child in node.children:
                if child in self.nodes:
                    self.nodes[child].parents.add(parent)

class Page:
    def __init__(self, url, title, content):
        self.url = url
        self.title = title
        self.content = content

    def __repr__(self):
        return f'Title = {self.title} \n URL = {self.url} \n Content = "{self.content}"'


class Scraper:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--headless=new")
        self.chrome_options.headless = True
        self.output_chat_list=['dead-page-title']
        # Load pre-trained tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.dag = DAG()

    def limit_tokens_from_string(self,string: str, model: str, limit: int) -> str:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except:
            encoding = tiktoken.encoding_for_model('gpt2')  # Fallback for others.

        encoded = encoding.encode(string)

        return encoding.decode(encoded[:limit])


    def openai_call(self,
        prompt: str,
        model: str = 'gpt-3.5-turbo',
        temperature: float = 0.2,
        max_tokens: int = 100,
    ):
        while True:
                if not model.lower().startswith("gpt-"):
                    # Use completion API
                    response = openai.Completion.create(
                        engine=model,
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        api_key=OPENAI_API_KEY,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                    )
                    return response.choices[0].text.strip()
                else:
                    
                    trimmed_prompt = self.limit_tokens_from_string(prompt, model, 6000 - max_tokens)

                    # Use chat completion API
                    client = OpenAI(api_key=OPENAI_API_KEY)


                    messages = [{"role": "system", "content": trimmed_prompt}]
                    response = client.chat.completions.create(
                        model=model,
                        #api_key=OPENAI_API_KEY,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        n=1,
                        stop=None,
                    )
                    return response.choices[0].message.content.strip()


    # Function to generate sentence embeddings
    def generate_embedding(self,sentence):
        input_ids = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)['input_ids']
        with torch.no_grad():
            outputs = self.model(input_ids)
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().detach()
        return embeddings


    def compute_similarity(self,target_sentence, sentences):
        target_embedding = self.generate_embedding(target_sentence)
        embeddings = [self.generate_embedding(sentence) for sentence in sentences]
        similarity_scores = []
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i, embedding in enumerate(embeddings):
            similarity_score = abs(cos(torch.tensor(target_embedding), torch.tensor(embedding)))
            similarity_scores.append(similarity_score)
        numpy_array = np.array([tensor.item() for tensor in similarity_scores])
        max_index = np.argmax(numpy_array)
        return similarity_scores, max_index

    def prompt_for_chat(self,objective):
        #input_searched_pages="\n".join(output_chat_list)
        input_chat=f"""Consider the following objective: '{objective}'.
                                \nGenerate a Github topic title that contains: 'pythorch forecast'.
                                \nFor example: 'How to forecast with PyTorch'. or 'Forecasting with PyTorch'.
                                \nThe output should be in a format like: https://www.google.com/search?q=forecasting+github+airport
                                \nAnswer: https://www.google.com/search?q="""

        output_chat = self.openai_call(input_chat.replace("    ",""), max_tokens=4000)
        output_chat=output_chat.replace(" ","+").replace("+","+").replace('"',"").replace('\n',"")
        output_chat="https://www.google.com/search?q=GitHub+"+output_chat
        return output_chat

    def chose_the_appropriate_project(self,repo_links,context_of_the_objective):
        readmes=[]
        for i,sdc in enumerate(repo_links):
            response = requests.get(sdc)
            html_content = response.text
            if "README.md" in html_content:
                redme_path=sdc+"/blob/main/README.md"
                readmes.append(self.add_file(redme_path,i,'md'))
        
        scores_similarity, max_readme_similarity_index =self.compute_similarity(context_of_the_objective,readmes)
        
        return repo_links[max_readme_similarity_index],readmes[max_readme_similarity_index]

    def add_file(self,page_link,counter,type='py'):
        if type=='py':
            f=open("codes/"+page_link.split("/")[-1][:-3]+".py", "w")
        elif type=='md':
            f=open("codes/readme_"+str(counter)+".md", "w")
        else:
            f=open("codes/"+page_link.split("/")[-1][:-6]+".ipynb", "w")
        response = requests.get(page_link.replace("github","raw.githubusercontent").replace("/blob","")).text
        #if  type!='md':
        f.write(response)
        f.close()
        print("Downloaded: ",page_link)
        return response

    def scrape(self,driver, page_link,num_downloads):
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, features="html.parser")
        repo_links = []
        if '.py' in page_source or '.ipynb' in page_source:

            pattern = r'(?<=href=")[^"]+\.py'
            py_files = re.findall(pattern, page_source)
            print("py_files")

            for x in list(set(py_files)):
                if "https://github.com/" in x:
                    web_page=x
                else:
                    web_page="https://github.com"+x

                print(web_page)
                if "comhttps" not in web_page:
                    answer=self.add_file(web_page,num_downloads,'py')
                time.sleep(5)
                num_downloads+=1


            pattern = r'(?<=href=")[^"]+\.ipynb'
            py_files = re.findall(pattern, page_source)

            for x in list(set(py_files)):
                if "https://github.com" in x:
                    web_page=x
                else:
                    web_page="https://github.com"+x

                print(web_page)
                if "comhttps" not in web_page:
                    answer=self.add_file(web_page,num_downloads,'ipynb')
                    time.sleep(5)
                num_downloads+=1

        if 'topics'in page_link:
            repo_elements = soup.find_all('h3', class_='f3', limit=1)
            for elem in repo_elements:
                link = elem.find('a', class_='Link text-bold wb-break-word').get('href')
                repo_links.append('https://github.com' + link)
        elif "google" in page_link:
            for a_tag in soup.find_all('a', href=True):
                if "https://github.com/" in a_tag['href'] and "=h" not in a_tag['href']:
                    repo_links.append(a_tag['href'])
                    print(a_tag['href'])
            #chose_the_appropriate_project(repo_links,objective)
            link_project, readme_chosed_project = self.chose_the_appropriate_project(repo_links,objective)
            f=open("codes/README.md", "w")
            f.write(link_project+"\n"+readme_chosed_project)
            repo_links=[link_project]
        else:
            for a_tag in soup.find_all('a', href=True):
                tag='https://github.com'+a_tag['href']
                if 'tree/' in tag and tag not in visited_pages and page_link in tag and '/.' not in tag:
                    repo_links.append(tag)

        return repo_links 


    def get_visit_pages(self):
        self.dag.populate_parents()
        nx_graph = self.dag.to_networkx()
        pos = nx.spring_layout(nx_graph)
        nx.draw(nx_graph, pos, with_labels=True, node_size=30, node_color="skyblue", font_size=3, font_color="black")
        plt.savefig("dag_image.png")
        plt.show()


    def crawl(self,page_link):
        try:
            thread_driver = webdriver.Chrome(options=self.chrome_options)
            if page_link not in visited_pages:
                visited_pages.append(page_link)
                if self.output_chat_list:
                    thread_driver.get(page_link)
                    local_links=self.scrape(thread_driver, page_link,len(visited_pages))
                    
                    if local_links is not None:
                        local_links = set(local_links)

                        for link in local_links.copy():
                            for visited_page in visited_pages:
                                if link == visited_page:
                                    local_links.remove(link)
                    
                    if len(local_links) != 0:
                        for link in local_links:
                            if link not in visited_pages:
                                self.dag.add_edge(page_link, link)
                                if '.py' not in page_link and 'ipynb' not in page_link:
                                    self.crawl(link)
        finally:
            print("QUITTING....")
            thread_driver.quit()



scraper_airport = Scraper()
scraper_airport.crawl(scraper_airport.prompt_for_chat(objective))
