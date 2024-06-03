import requests
import base64
from dotenv import load_dotenv
from flask import Flask,request,jsonify
from flask_cors import CORS, cross_origin
import os
import json
import PyPDF2
import nltk
import re
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer, util
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate as LangchainPromptTemplate
from langchain_core.prompts import PromptTemplate as LangchainCorePromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI
# from getpass import getpass
# OPENAI_API_KEY = getpass()
# os.environ["OPENAI_API_KEY"]
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
        quotient=len(lines)
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
        prompt = LangchainPromptTemplate(template=template, input_variables=["contextLine"])
        llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(repo_id="voidful/context-only-question-generator", model_kwargs={"temperature":0.5}))
        questions=questions+(llm_chain.run(line),)
    return list(questions)

def extractNumbersFromString(s):
    # Use regular expression to find all numbers (integers and floats)
    numbers = re.findall(r'-?\d+\.?\d*', s)
    # Convert extracted strings to float or int
    numbers = [float(num) if '.' in num else int(num) for num in numbers]
    return numbers

def evaluateSingleResponse(question_data, response):
    print(f'question_data: {question_data}')
    print(f'response: {response}')
    if len(question_data['options']) > 0:
        options = ", ".join(question_data['options'])
        question = question_data['question']
        template = """Response to the question: {question} with options {options} is {response}. Provide only the number 1 if the answer is correct, otherwise 0. Do not include any explanation."""
        prompt = LangchainCorePromptTemplate.from_template(template)
        llm = OpenAI(openai_api_key="sk-proj-7Ss1AMrKAK3vnV6k9Jt2T3BlbkFJ38GBoCSOqIVupfZAiByI")
        llm_chain = prompt | llm
        input_variables = {
            "question": question,
            "options": options,
            "response": response
        }
        result = llm_chain.invoke(input_variables)
        result = result.replace("\n", "")
        result = extractNumbersFromString(result)[0]
        return float(result)*100
    else:
        question = question_data['question']    
        template = """Response to the question: {question} is {response}. Provide only a score between 0 and 100. Do not include any explanation."""
        prompt = LangchainCorePromptTemplate.from_template(template)
        llm = OpenAI(openai_api_key="sk-proj-7Ss1AMrKAK3vnV6k9Jt2T3BlbkFJ38GBoCSOqIVupfZAiByI")
        llm_chain = prompt | llm
        input_variables = {
            "question": question,
            "response": response
        }
        result = llm_chain.invoke(input_variables)
        result = result.replace("\n", "")
        result = extractNumbersFromString(result)[0]
        return float(result)

@app.route('/evaluation', methods=['POST'])
def evaluation():
    # try:
    #     question=request.json['question']
    #     response=request.json['response']
    #     if(len(question['options'])>0):
    #         options=", ".join(question['options'])
    #         question=question.question
    #         template = """Response to the question: {question} with options {options}  is {response}. Now give 1 in response if the answer is correct otherwise 0."""
    #         prompt = PromptTemplate.from_template(template)
    #         llm = OpenAI(openai_api_key="sk-proj-7Ss1AMrKAK3vnV6k9Jt2T3BlbkFJ38GBoCSOqIVupfZAiByI")
    #         llm_chain = prompt | llm
    #         input_variables = {"question": question, "options": options, "response": response}
    #         result=llm_chain.invoke(question)
    #         return jsonify({'response': result})
    #     else:
    #         question=question.question
    #         template = """Response to the question: {question} is {response}. Now give score to this response between 0 to 1."""
    #         prompt = PromptTemplate.from_template(template)
    #         llm = OpenAI(openai_api_key="sk-proj-7Ss1AMrKAK3vnV6k9Jt2T3BlbkFJ38GBoCSOqIVupfZAiByI")
    #         llm_chain = prompt | llm
    #         input_variables = {"question": question, "response": response}
    #         result=llm_chain.invoke(input_variables)
    #         return jsonify({'response': result})
    try:
        responses=request.json['responses']
        print(f'responses: {responses}')
        finalResult=[]
        # finalDict={}
        for singleResponse in responses:
            finalDict={}
            finalDict['question']=singleResponse['question']['question']
            finalDict['response']=singleResponse['response']
            finalDict['score']=evaluateSingleResponse(singleResponse['question'], singleResponse['response'])
            finalResult.append(finalDict)
        return jsonify(finalResult)
    except Exception as e:
        return jsonify(str(e))

#   APIs
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
        print(f'linesList: {linesList}')
        result=generateQuestionsFromContext(linesList)
        return jsonify(result)
    except Exception as e:
        return jsonify(str(e))

@app.route('/evaluationresponse', methods=['POST'])
def evaluationresponse():
    try:
        user_responses = request.json['user_responses']
        
        if not user_responses:
            return jsonify("User responses cannot be empty"), 400
        
        evaluation_results = []
        for user_response in user_responses:
            best_match_score = 0
            best_match_content = ""
            
            for file_content_line, _ in staticFilesContent.items():
                sentences = [user_response, file_content_line]
                embeddings = model.encode(sentences, convert_to_tensor=True)
                similarity_score = util.cos_sim(embeddings[0], embeddings[1])[0].item()
                
                if similarity_score > best_match_score:
                    best_match_score = similarity_score
                    best_match_content = file_content_line
            
            evaluation_results.append({
                "user_response": user_response,
                "best_match_content": best_match_content,
                "similarity_score": best_match_score
            })
        
        return jsonify(evaluation_results)
    except Exception as e:
        return jsonify(str(e)), 500

if __name__ == "__main__":
    load_dotenv()
    filepaths=collectFilepaths()
    extractedContent=[]
    for filepath in filepaths:
        if(filepath.endswith(".pdf")):
            extractedContent=extractedContent+extractPDF(filepath).split('. ')
        elif(filepath.endswith(".txt")):
            extractedContent=extractedContent+extractTXT(filepath).split('. ')
    for extractedContentLine in extractedContent:
        staticFilesContent[extractedContentLine]=0
    app.run(debug=True, port=8000)