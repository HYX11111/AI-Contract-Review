#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import transformers
import torch
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, TfidfRetriever, FilterRetriever, DensePassageRetriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from docx import Document


# In[2]:


import import_ipynb
from Load_and_Print import read_qas, print_result


# In[3]:


def load_test_data(file):
    with open(file) as jfile:
        data = json.load(jfile) 
        samples = []
    for i in range (0,10):
        contract = "".join(data["data"][i]["paragraphs"][0]["context"].split("\n"))
        samples.append(contract)
    return samples


# In[4]:


def read_qas(file):
    qas_doc = Document(file)
    questions = []
    for para in qas_doc.paragraphs:
        question = {}
        k, v = para.text.split(":", 1)
        question[k.strip()] = v.strip()
        questions.append(question)
    return questions


# In[5]:


def test_model1(qs, c):
    model = transformers.AutoModelForQuestionAnswering.from_pretrained("akdeniz27/roberta-large-cuad")
    tokenizer = transformers.AutoTokenizer.from_pretrained("akdeniz27/roberta-large-cuad")
    qa = transformers.pipeline("question-answering", model=model, tokenizer=tokenizer)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        querry = list(qs[j].values())[0]
        res = qa({
        "question": querry,
        "context": c
        })
        print(res)
        ans.append(res)
    return ans


# In[6]:


def test_model2(qs, c):
    model = transformers.AutoModelForQuestionAnswering.from_pretrained("Rakib/roberta-base-on-cuad")
    tokenizer = transformers.AutoTokenizer.from_pretrained("Rakib/roberta-base-on-cuad")
    qa = transformers.pipeline('question-answering', model=model, tokenizer=tokenizer)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        querry = list(qs[j].values())[0]
        res = qa({
        "question": querry,
        "context": c
        })
        print(res)
        ans.append(res)
    return ans


# In[26]:


def test_model3(qs, c, doc_store):
    retriever = BM25Retriever(doc_store) 
    reader = FARMReader(model_name_or_path='Rakib/roberta-base-on-cuad',
                        context_window_size=1500,
                        use_gpu=True)
    qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        query = list(qs[j].values())[0]
        res = qa.run(
            query=query, params={"Reader": {"top_k": 1},"Retriever": {"top_k": 1}}
        )
        print_answers(res, details = "minimum")
        ans.append(res)
    return ans


# In[8]:


def test_model4(qs, c, doc_store):
    retriever = BM25Retriever(doc_store)
    reader = FARMReader(model_name_or_path='akdeniz27/roberta-large-cuad',
                        
                        context_window_size=1500,
                        use_gpu=True)
    qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        query = list(qs[j].values())[0]
        res = qa.run(
            query=query, params={"Reader": {"top_k": 1},"Retriever": {"top_k": 1}}
        )
        print_answers(res, details = "minimum")
        ans.append(res)
    return ans


# In[9]:


def test_model5(qs, c, doc_store):
    retriever = TfidfRetriever(doc_store) 
    reader = FARMReader(model_name_or_path='Rakib/roberta-base-on-cuad',
                        context_window_size=1500,
                        use_gpu=True)
    qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        query = list(qs[j].values())[0]
        res = qa.run(
            query=query, params={"Reader": {"top_k": 1},"Retriever": {"top_k": 1}}
        )
        print_answers(res, details = "minimum")
        ans.append(res)
    return ans


# In[10]:


def test_model6(qs, c, doc_store):
    retriever = FilterRetriever(doc_store) 
    reader = FARMReader(model_name_or_path='Rakib/roberta-base-on-cuad',
                        context_window_size=1500,
                        use_gpu=True)
    qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        query = list(qs[j].values())[0]
        res = qa.run(
            query=query, params={"Reader": {"top_k": 1},"Retriever": {"top_k": 1}}
        )
        print_answers(res, details = "minimum")
        ans.append(res)
    return ans


# In[11]:


def test_model7(qs, c, doc_store):
    retriever = BM25Retriever(doc_store) 
    reader = FARMReader(model_name_or_path='deepset/bert-base-cased-squad2',
                        context_window_size=1500,
                        use_gpu=True)
    qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        query = list(qs[j].values())[0]
        res = qa.run(
            query=query, params={"Reader": {"top_k": 1},"Retriever": {"top_k": 1}}
        )
        print_answers(res, details = "minimum")
        ans.append(res)
    return ans


# In[12]:


def test_model8(qs, c, doc_store):
    retriever = DensePassageRetriever(
        document_store=doc_store,
        query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
        passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
        use_gpu=True, embed_title=True)
    doc_store.update_embeddings(retriever=retriever)
    retriever = BM25Retriever(doc_store) 
    reader = FARMReader(model_name_or_path='Rakib/roberta-base-on-cuad',
                        context_window_size=1500,
                        use_gpu=True)
    qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)
    ans = []
    for j in range(0, len(qs)):
        print (f"qs{j+1}")
        query = list(qs[j].values())[0]
        res = qa.run(
            query=query, params={"Reader": {"top_k": 1},"Retriever": {"top_k": 1}}
        )
        print_answers(res, details = "minimum")
        ans.append(res)
    return res


# In[13]:


def set_doc_store(c, i):
    doc_store = ElasticsearchDocumentStore(
        host='localhost',
        username='elastic', password='hrAezzG2su2WPqvOu8J9',
        index= str(i)
    )
    data_json = [
    {
        'content':c,
        'meta':{
            'source': f"contract {str(i)}"
            }
        }
    ]
    doc_store.write_documents(data_json, index = str(i))
    return doc_store


# In[14]:


qas = read_qas("./Questions.docx")
samples = load_test_data("./test.json")


# In[15]:


def test1():
    print("Model1 Result")
    for i in range(0, 5):
        context = samples[i]
        print(f"contract{i+1}")
        test_model1(qas, context)


# In[16]:


def test2():
    print("Model2 Result")
    for i in range(0,5):
        context = samples[i]
        print(f"contract{i+1}")
        test_model2(qas, context)


# In[17]:


def test3():
    print("Model3 Result")
    for i in range(0, 5):
        context = samples[i]
        print(f"contract{i+1}")
        doc_store = set_doc_store(context, i+20)
        test_model3(qas, context, doc_store)
        doc_store.delete_index(str(i+20))


# In[18]:


def test4():
    print("Model4 Result")
    for i in range(0, 5):
        context = samples[i]
        print(f"contract{i+1}")
        doc_store = set_doc_store(context, i+30)
        test_model4(qas, context, doc_store)
        doc_store.delete_index(str(i+30))


# In[19]:


def test5():
   print("Model5 Result")
   for i in range(0, 5):
       context = samples[i]
       print(f"contract{i+1}")
       doc_store = set_doc_store(context, i+40)
       test_model5(qas, context, doc_store)
       doc_store.delete_index(str(i+40))


# In[20]:


def test6():    
    print("Model6 Result")
    for i in range(0, 5):
        context = samples[i]
        print(f"contract{i+1}")
        doc_store = set_doc_store(context, i+50)
        test_model6(qas, context, doc_store)
        doc_store.delete_index(str(i+50))


# In[21]:


def test7():
    print("Model7 Result")
    for i in range(0, 5):
        context = samples[i]
        print(f"contract{i+1}")
        doc_store = set_doc_store(context, i+60)
        test_model7(qas, context, doc_store)
        doc_store.delete_index(str(i+60))


# In[33]:


def test8():
    print("Model8 Result")
    for i in range(0, 5):
        context = samples[i]
        print(f"contract{i+1}")
        doc_store = set_doc_store(context, i+70)
        test_model8(qas, context, doc_store)
        doc_store.delete_index(str(i+70))


# In[23]:


if __name__ == '__main__':
    test1()


# In[24]:


if __name__ == '__main__':
    test2()


# In[27]:


if __name__ == '__main__':
    test3()


# In[28]:


if __name__ == '__main__':
    test4()


# In[29]:


if __name__ == '__main__':   
    test5()


# In[30]:


if __name__ == '__main__':
    test6()


# In[31]:


if __name__ == '__main__':
    test7()


# In[34]:


if __name__ == '__main__':
    test8()


# In[ ]:




