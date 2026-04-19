import os
import json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ---------------------------------------------------------
# STATE DEFINITION
# ---------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    pedagogy_context: str
    web_context: str

# ---------------------------------------------------------
# RAG STORE SETUP
# ---------------------------------------------------------
PEDAGOGY_DOCS = [
    "To improve a 'Hard' question, ensure that the distractors address specific common misconceptions.",
    "If discrimination index is low, the question may be poorly worded or confusing.",
    "Bloom's Taxonomy suggests that 'Evaluate' and 'Create' cognitive levels naturally correlate with higher difficulty.",
    "For 'Medium' questions with high standard deviation, standardizing the vocabulary can reduce cognitive load."
]

def init_vectorstore():
    # Only loads if needed to save time
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=t) for t in PEDAGOGY_DOCS]
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 2})

try:
    retriever = init_vectorstore()
except Exception as e:
    retriever = None

# Initialize DuckDuckGo tool
ddg_search = None

# ---------------------------------------------------------
# LANGGRAPH NODES
# ---------------------------------------------------------
def retrieve_pedagogy_node(state: AgentState):
    """Retrieves relevant pedagogy guidelines."""
    user_query = state['messages'][-1].content
    if retriever:
        docs = retriever.get_relevant_documents(user_query)
        context = "\n".join([d.page_content for d in docs])
    else:
        context = "Pedagogy retrieval unavailable."
    return {"pedagogy_context": context}

def web_search_node(state: AgentState):
    """Searches web for recent educational best practices."""
    user_query = state['messages'][-1].content
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text("teaching assessment design: " + user_query[:50], max_results=2):
                results.append(r.get('body', ''))
        results_str = "\n".join(results)
    except Exception as e:
        results_str = ""
    return {"web_context": results_str[:1500]} 

def generate_recommendations_node(state: AgentState):
    """Generates a conversational analysis using the LLM."""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        api_key=os.environ.get("GROQ_API_KEY", "")
    )
    
    system_msg = """You are an expert Educational Assessment Designer.
The user will provide details about a specific exam question and its performance metrics.

Generate a simple, genuine, and easy-to-understand analysis. Keep it concise, especially for short answer questions.

You MUST return EXACTLY a JSON object (without any markdown formatting or backticks) with these exact three keys:
{{
    "reasoning": "A simple explanation of why the question has its predicted difficulty and its overall quality.",
    "learning_gaps": "A concise analysis of what students are misunderstanding or struggling with.",
    "recommendations": "Clear, direct, and actionable advice to improve the question."
}}

Pedagogy Guidelines (from Knowledge Base):
{pedagogy}

Latest Web Context:
{web}
"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_msg),
        ("placeholder", "{messages}")
    ])
    
    chain = prompt | llm
    
    try:
        res = chain.invoke({
            "messages": state['messages'],
            "pedagogy": state.get('pedagogy_context', ''),
            "web": state.get('web_context', '')
        })
        # Clean up any potential markdown block wrappers from the completion
        raw = res.content
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        res = AIMessage(content=raw)
    except Exception as e:
        res = AIMessage(content=json.dumps({
            "reasoning": f"Error: {str(e)}",
            "learning_gaps": "API configuration issue.",
            "recommendations": "Check GROQ_API_KEY."
        }))
        
    return {"messages": [res]}

# ---------------------------------------------------------
# GRAPH BUILDER
# ---------------------------------------------------------
def create_agent_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("retrieve", retrieve_pedagogy_node)
    workflow.add_node("search", web_search_node)
    workflow.add_node("generate", generate_recommendations_node)
    
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "search")
    workflow.add_edge("search", "generate")
    workflow.add_edge("generate", END)
    
    app = workflow.compile()
    return app

def run_agentic_workflow(user_query: str, api_key: str):
    """Wrapper that mimics the conversational invocation pattern."""
    if api_key:
        os.environ["GROQ_API_KEY"] = api_key
        
    agent = create_agent_graph()
    
    # Invoking the graph using the messages schema
    response = agent.invoke({
        "messages": [
            {
                "role": "user", 
                "content": user_query
            }
        ]
    })
    
    raw_answer = response["messages"][-1].content
    return raw_answer
