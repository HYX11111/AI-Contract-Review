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
import Test_models as test_models
import Test_process_contracts as test_process
from Load_and_Print import read_qas, print_result


# In[3]:


##Choose model 3 for retrivers and readers, and choose model 1 for processing context.


# In[4]:


def review_contract():
    qas = read_qas("./Questions.docx")
    
    for i in range (0,10):
        index = i + 1
        input_dir = f"./Agreements/Test Agreement {index}.docx"
        print(f"Contract{index}")
        contract = test_process.process_contract1(input_dir)
        doc_store = test_models.set_doc_store(contract, index)
        ans = test_models.test_model3(qas, contract, doc_store)
        doc_store.delete_index(str(index))
        print_result(f"Contract{index}", ans)


# In[5]:


if __name__ == '__main__':
    review_contract()


# In[ ]:




