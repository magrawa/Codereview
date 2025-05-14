import google.generativeai as genai
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI

GOOGLE_API_KEY = 'AIzaSyDzyFmJy9BdqeYaHIvQhsnXP9Hr4EAFWtxk'
genai.configure(api_key=GOOGLE_API_KEY)

model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)


def llm(x):
    return model.invoke(x).content

class GraphState(TypedDict):
    feedback: Optional[str] = None
    history: Optional[str] = None
    code: Optional[str] = None
    specialization: Optional[str]=None
    rating: Optional[str] = None
    iterations: Optional[int]=None
    code_compare: Optional[str]=None
    actual_code: Optional[str]=None


workflow = StateGraph(GraphState)

reviewer_start= "You are Code reviewer specialized in {}.\
You need to review the given code following PEP8 guidelines and potential bugs\
and point out issues as bullet list.\
Code:\n {}"

coder_start = "You are a Coder specialized in {}.\
Improve the given code given the following guidelines. Guideline:\n {} \n \
Code:\n {} \n \
Output just the improved code and nothing else."

rating_start = "Rate the skills of the coder on a scale of 10 given the Code review cycle with a short reason.\
Code review:\n {} \n "

code_comparison = "Compare the two code snippets and rate on a scale of 10 to both. Dont output the codes.Revised Code: \n {} \n Actual Code: \n {}"

classify_feedback = "Are all feedback mentioned resolved in the code? Output just Yes or No.\
Code: \n {} \n Feedback: \n {} \n"

def handle_reviewer(state):
    history = state.get('history', '').strip()
    code = state.get('code', '').strip()
    specialization = state.get('specialization','').strip()
    iterations = state.get('iterations')
    
    print("Reviewer working...")
    
    feedback = llm(reviewer_start.format(specialization,code))
    
    return {'history':history+"\n REVIEWER:\n"+feedback,'feedback':feedback,'iterations':iterations+1}

def handle_coder(state):
    history = state.get('history', '').strip()
    feedback = state.get('feedback', '').strip()
    code =  state.get('code','').strip()
    specialization = state.get('specialization','').strip()
    
    print("CODER rewriting...")
    
    code = llm(coder_start.format(specialization,feedback,code))
    return {'history':history+'\n CODER:\n'+code,'code':code}

def handle_result(state):
    print("Review done...")
    
    history = state.get('history', '').strip()
    code1 = state.get('code', '').strip()
    code2 = state.get('actual_code', '').strip()
    rating  = llm(rating_start.format(history))
    
    code_compare = llm(code_comparison.format(code1,code2))
    return {'rating':rating,'code_compare':code_compare}

workflow.add_node("handle_reviewer",handle_reviewer)
workflow.add_node("handle_coder",handle_coder)
workflow.add_node("handle_result",handle_result)

def deployment_ready(state):
    deployment_ready = 1 if 'yes' in llm(classify_feedback.format(state.get('code'),state.get('feedback'))) else 0
    total_iterations = 1 if state.get('iterations') > 5 else 0
    return "handle_result" if  deployment_ready or total_iterations else "handle_coder" 


workflow.add_conditional_edges(
    "handle_reviewer",
    deployment_ready,
    {
        "handle_result": "handle_result",
        "handle_coder": "handle_coder"
    }
)

workflow.set_entry_point("handle_reviewer")
workflow.add_edge('handle_coder', "handle_reviewer")
workflow.add_edge('handle_result', END)

specialization = 'python'
problem = 'Generate code to train a Regression ML model using a tabular dataset following required preprocessing steps.'
code = llm(problem)

app = workflow.compile()
conversation = app.invoke(
    {
        "history": code,
        "code": code,
        "actual_code": code,
        "specialization": specialization,
        "iterations": 0
    },
    {"recursion_limit": 100}
)
print(conversation['code'])
print(conversation['history'])
print(conversation['feedback'])
print(conversation['rating'])
print(conversation['code_compare'])
