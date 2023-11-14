from flask import Flask,request,jsonify
import os
import openai
import json
import spacy
from spacy.matcher import PhraseMatcher

# load default skills data base
from skillNer.general_params import SKILL_DB
# import skill extractor
from skillNer.skill_extractor_class import SkillExtractor

# init params of skill extractor
nlp = spacy.load("en_core_web_lg")
# init skill extractor
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    prompt=request.json['prompt']
    mcqs=request.json['mcqs']
    questions=request.json['questions']
    problems=request.json['problems']

        
    skills=""
    try:
        annotations = skill_extractor.annotate(prompt)
        full_matches=annotations['results']['full_matches']
        for match in full_matches:
            skills+=f"{match['doc_node_value']}, "
        ngram_scored=annotations['results']['ngram_scored']
        for match in ngram_scored:
            skills+=f"{match['doc_node_value']}, "
        # print(f"annotations: {annotations}")
    except:
        return jsonify({"message":"error occured extracting skills from prompt"})

    content=f'Generate a skill test related to the skills: {skills}. There should be {mcqs} MCQs, {questions} theoretical questions, {problems} problem solving questions. Your response should be an array of objects containing each question. Object of MCQ must be like {{"q":"this is the question","o1":"option1","o2":"option2","o3":"option3","o4":"option4"}} and other questions objects should be just like {{"q":"this is the question"}}. Just return me array of these objects nothing else !'

    chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content":content }])

    response=chat_completion.choices[0].message.content
    print(f"response: {response}")
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True, port=8000)