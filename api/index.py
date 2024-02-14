from dotenv import load_dotenv
from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
import os
import json
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)
cors = CORS(app)

staticFilesContent={}
model = SentenceTransformer('all-MiniLM-L6-v2')

def collectFilepaths():
    filepaths=[]
    staticFolder=os.path.abspath('./api/static')
    for root, _, files in os.walk(staticFolder):
        for filename in files:
            filepath = os.path.join(root, filename)
            if any(filepath.endswith(ext) for ext in (".pdf, .txt")):
                filepaths.append(filepath)
    return filepaths

def extractPDF(pdfpath):
    text = ""
    with open(pdfpath, "rb") as pdfFile:
        pdfReader = PyPDF2.PdfReader(pdfFile)
        for pageNum in range(len(pdfReader.pages)):  
            page = pdfReader.pages[pageNum]
            text += page.extract_text()
    return text

def extractTXT(txtpath):
    with open(txtpath, "r", encoding="utf-8") as txtfile:
        text = txtfile.read()
    return text

def extractContextFromContent(prompt, numberOfLines):
    for fileContentLine in staticFilesContent.keys():
        sentences = [prompt, fileContentLine]

        embeddings = model.encode(sentences, convert_to_tensor=True)
        similarity_0_1 = util.cos_sim(embeddings[0], embeddings[1])[0].item()
        staticFilesContent[fileContentLine]=similarity_0_1
        # print(f"Sentence 0: {sentences[0]}")
        # print(f"Sentence 1: {sentences[1]}")
        # print(f"Similarity: {similarity_0_1}")
    # print(f'dictionary updated: {staticFilesContent}')
    print(f'type: {type(staticFilesContent)}')
    return sorted(staticFilesContent.items(), key=lambda x: x[1], reverse=True)[:numberOfLines]




def getSeparateLines(context, numberOfQuestions):
    lines=context.split('. ')
    if len(lines)<=numberOfQuestions:
        return lines
    else:
        quotient=len(lines)//numberOfQuestions
        finalTuple=()
        for i in range(numberOfQuestions):
            listTuple=()
            for j in range(quotient):
                listTuple=listTuple+(lines[i*quotient+j],)
            finalTuple=finalTuple+(". ".join(listTuple),)
        return list(finalTuple)

def generateQuestionsFromContext(linesList):
    
    questions=()
    for line in linesList:
        template = """{contextLine}"""

        prompt = PromptTemplate(template=template, input_variables=["contextLine"])
        llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="voidful/context-only-question-generator", model_kwargs={"temperature":0.5}))
        questions=questions+(llm_chain.run(line),)
    return list(questions)

@app.route('/prompt', methods=['POST'])
def questionsFromPrompt():
    try:
        prompt=request.json['prompt']
        numberOfQuestions=int(request.json['questions'])
        extractedContext=extractContextFromContent(prompt, numberOfQuestions)
        result=generateQuestionsFromContext(extractedContext)
        return jsonify(result)
    except Exception as e:
        return jsonify(str(e))
    
@app.route('/context', methods=['POST'])
def questionsFromContext():
    try:
        context=request.json['prompt']
        numberOfQuestions=int(request.json['questions'])
        linesList=getSeparateLines(context, numberOfQuestions)
        result=generateQuestionsFromContext(linesList)
        return jsonify(result)
    except Exception as e:
        return jsonify(str(e))

if __name__ == "__main__":
    load_dotenv()
    filepaths=collectFilepaths()
    extractedContent=[]
    for filepath in filepaths:
        if(filepath.endswith(".pdf")):
            extractedContent=extractPDF(filepath).split('. ')
        elif(filepath.endswith(".txt")):
            extractedContent=extractTXT(filepath).split('. ')
    for extractedContentLine in extractedContent:
        staticFilesContent[extractedContentLine]=0
    # print(f"context dictionary: {staticFilesContent}")
    app.run(debug=True, port=8000)