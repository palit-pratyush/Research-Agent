import streamlit as st
import arxiv
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from datetime import datetime
import logging
import json
from dotenv import load_dotenv
import time
import pandas as pd

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    .st-emotion-cache-se9ihy {
        display: flex;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #E2E8F0;
        min-height: 100vh;
        line-height: 1.6;
    }

    .main-header {
        padding: 2rem;
        margin-bottom: 4rem;
        text-align: center;
    }
    
    .main-header h1 {
        font-size: 3.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #8B5CF6 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
    }
    
    .main-header p {
        font-size: 1.5rem;
        color: #94A3B8;
        font-weight: 400;
        max-width: 600px;
        margin: 0 auto;
    }

    .stSidebar {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        color: #94A3B8; 
    }

    .stats-container {
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(139, 92, 246, 0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: bounceIn 1s ease-out;
    }
    
    @keyframes bounceIn {
        0% { transform: scale(0.3); opacity: 0; }
        50% { transform: scale(1.05); }
        70% { transform: scale(0.9); }
        100% { transform: scale(1); opacity: 1; }
    }
    
    .stats-container h4 {
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.2rem;
    }
    
    .stats-container p {
        margin: 0.5rem 0;
        font-size: 0.95rem;
        opacity: 0.9;
    }

    .stButton button {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        text-decoration: none;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    .stButton button:hover {
        background: rgba(139, 92, 246, 0.3);
        transform: translateY(-1px);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.2); 
    }

    .stButton button:active {
        transform: translateY(0);
    }

    .stForm {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 0.75rem;
        border: 1px solid rgba(71, 85, 105, 0.5);
        backdrop-filter: blur(10px);
        display: flex;
        align-items: center;
        gap: 1rem;
        transition: all 0.3s ease;
    }

    .stForm:hover {
        border-color: rgba(139, 92, 246, 0.5);
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.1);
    }

    .stForm:focus-within {
        border-color: #8B5CF6;
        box-shadow: 0 0 20px rgba(139, 92, 246, 0.2);
    }
            
    .stTextArea > div > div > textarea {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(71, 85, 105, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 3px;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        color: #E2E8F0;
    }
            
    .stTextInput > div > div > input {
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(71, 85, 105, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 3px;
        padding: 1rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        color: #E2E8F0;
    }


    .stFormSubmitButton button {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        text-decoration: none;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    .stFormSubmitButton button:hover {
        background: rgba(139, 92, 246, 0.3);
        transform: translateY(-1px);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.2); 
    }

    .stFormSubmitButton button:active {
        transform: translateY(0);
    }

    .paper-card {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(71, 85, 105, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .paper-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        border-color: rgba(139, 92, 246, 0.3);
    }
    
    .relevance-score {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .stCheckbox {
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        margin-top: 10px;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .stCheckbox:hover {
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        transform: translateY(-2px);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }

    .stExpander {
        background: rgba(139, 92, 246, 0.1);
        border-left: 4px solid rgba(139, 92, 246, 0.2);
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .link-button {
        display: inline-block;
        padding: 8px 16px;
        background: rgba(139, 92, 246, 0.1);
        color: #c4b5fd;
        text-decoration: none;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .link-button:hover {
        background: rgba(139, 92, 246, 0.3);
        transform: translateY(-1px);
        color: #c4b5fd;
        border: 1px solid rgba(139, 92, 246, 0.2); 
    }

    .chat-container {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        font-family: Georgia, 'Times New Roman', Times, serif;
    }
    
    .loading-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 2rem 0;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
            

    .stSelectbox > div > div{
            background: rgba(139, 92, 246, 0.1);
            color: #c4b5fd;
            border: 1px solid rgba(139, 92, 246, 0.2); 
    }
            
    
               
</style>
""", unsafe_allow_html=True)


if 'papers' not in st.session_state:
    st.session_state.papers = []
if 'selected_papers' not in st.session_state:
    st.session_state.selected_papers = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'search_performed' not in st.session_state:
    st.session_state.search_performed = False


@st.cache_resource
def load_models():
    """Load sentence transformer model"""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_llm():
    """Initialize LLM"""
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        st.error("⚠️ GROQ_API_KEY environment variable is required")
        st.stop()
    return ChatGroq(
        groq_api_key=api_key,
        model_name="llama3-8b-8192",  
        temperature=0.3
    )


sentence_model = load_models()

class PaperSummarizer:
    def __init__(self):
        self.llm = get_llm()
        self.summary_prompt = ChatPromptTemplate.from_template(
            """
            Please provide a comprehensive summary of this research paper in exactly 3 paragraphs:
            
            Title: {title}
            Abstract: {abstract}
            
            Format your response as follows:
            1. First paragraph: Main research question, methodology, and approach
            2. Second paragraph: Key findings and results
            3. Third paragraph: Significance, implications, and potential applications
            
            Keep each paragraph 3-4 sentences long and focus on the most important aspects.
            """
        )
        
        self.qa_prompt = ChatPromptTemplate.from_template(
            """
            Based on the following research papers and their summaries, please answer the user's question:
            
            Papers:
            {papers_context}
            
            User Question: {question}
            
            Please provide a comprehensive answer based on the information from these papers. 
            If the papers don't contain enough information to fully answer the question, 
            mention what aspects you cannot address based on the available papers.
            """
        )
    
    def summarize_paper(self, title, abstract):
        try:
            messages = self.summary_prompt.format_messages(
                title=title,
                abstract=abstract
            )
            response = self.llm.invoke(messages)  
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing paper: {e}")
            return f"Error generating summary: {str(e)}"
    
    def answer_question(self, question, papers_context):
        try:
            messages = self.qa_prompt.format_messages(
                question=question,
                papers_context=papers_context
            )
            response = self.llm.invoke(messages)  
            return response.content
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error generating answer: {str(e)}"


summarizer = PaperSummarizer()

# def fetch_arxiv_papers(query, max_results=10):
#     """Fetch papers from ArXiv API"""
#     try:
#         client = arxiv.Client(    
#             page_size=10,
#             delay_seconds=3.0,
#             num_retries=3
#         )
#         search = arxiv.Search(
#             query=query,
#             max_results=max_results,
#             sort_by=arxiv.SortCriterion.Relevance
#         )
        
#         papers = []
#         for result in client.results(search):
#             paper = {
#                 'id': result.entry_id.split('/')[-1],
#                 'title': result.title,
#                 'authors': [author.name for author in result.authors],
#                 'abstract': result.summary,
#                 'published': result.published.strftime('%Y-%m-%d'),
#                 'url': result.entry_id,
#                 'pdf_url': result.pdf_url
#             }
#             papers.append(paper)
        
#         return papers
#     except Exception as e:
#         logger.error(f"Error fetching papers: {e}")
#         return []

def fetch_arxiv_papers(query, max_results=10, max_attempts=3):
    """Fetch papers from ArXiv API with exponential backoff"""
    for attempt in range(max_attempts):
        try:
            client = arxiv.Client(page_size=10, delay_seconds=5.0, num_retries=1)
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            papers = []
            for result in client.results(search):
                papers.append({
                    'id': result.entry_id.split('/')[-1],
                    'title': result.title,
                    'authors': [a.name for a in result.authors],
                    'abstract': result.summary,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'source': 'arXiv'
                })
            return papers
        except Exception as e:
            if '429' in str(e) and attempt < max_attempts - 1:
                wait = (2 ** attempt) * 8  # 8, 16s
                logger.warning(f"arXiv rate limited, waiting {wait}s (attempt {attempt+1}/{max_attempts})")
                time.sleep(wait)
                continue
            logger.error(f"arXiv fetch failed: {e}")
            return []
    return []


def fetch_semantic_scholar_papers(query, max_results=10):
    """Fallback: fetch papers from Semantic Scholar API"""
    try:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,authors,year,publicationDate,externalIds,openAccessPdf,url"
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        papers = []
        for item in data.get("data", []):
            if not item.get("abstract"):
                continue  # skip papers with no abstract, summarizer needs it
            arxiv_id = (item.get("externalIds") or {}).get("ArXiv")
            papers.append({
                'id': item.get("paperId"),
                'title': item.get("title", "Untitled"),
                'authors': [a.get("name", "Unknown") for a in item.get("authors", [])],
                'abstract': item.get("abstract", ""),
                'published': item.get("publicationDate") or f"{item.get('year', 'Unknown')}-01-01",
                'url': item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId')}",
                'pdf_url': (item.get("openAccessPdf") or {}).get("url") or (
                    f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else item.get("url", "")
                ),
                'source': 'Semantic Scholar'
            })
        return papers
    except Exception as e:
        logger.error(f"Semantic Scholar fetch failed: {e}")
        return []


def fetch_papers(query, max_results=10):
    """Fetch papers from arXiv, falling back to Semantic Scholar if arXiv fails"""
    papers = fetch_arxiv_papers(query, max_results)
    if papers:
        return papers

    logger.warning("arXiv returned nothing — falling back to Semantic Scholar")
    st.info("⚠️ ArXiv is rate-limited right now — showing results from Semantic Scholar instead.")
    return fetch_semantic_scholar_papers(query, max_results)

def calculate_relevance_scores(query, papers):
    """Calculate relevance scores using cosine similarity"""
    try:
        query_embedding = sentence_model.encode([query])
        paper_texts = [f"{paper['title']} {paper['abstract']}" for paper in papers]
        paper_embeddings = sentence_model.encode(paper_texts)
        similarities = cosine_similarity(query_embedding, paper_embeddings)[0]
        
        for i, paper in enumerate(papers):
            paper['relevance_score'] = float(similarities[i])
        
        papers.sort(key=lambda x: x['relevance_score'], reverse=True)
        return papers
    except Exception as e:
        logger.error(f"Error calculating relevance: {e}")
        return papers

def display_paper_card(paper, index):
    """Display a paper card with modern styling"""
    with st.container():
        st.markdown(f"""
        <div class="paper-card fade-in">
            <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
                <h3 style="margin-bottom: 0.5rem; color: #E2E8F0; font-weight: 600; font-size: 1.5rem; line-height: 1.4;">{paper['title']}</h3>
                <span class="relevance-score">{paper['relevance_score']*100:.2f}%</span>
            </div>
            <p style="color: #8B5CF6; margin: 0.5rem 0; font-size: 0.9rem; font-weight: 500;">
                <strong>Authors:</strong> {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}
            </p>
            <p style="color: #8B5CF6; margin: 0.5rem 0; font-size: 0.9rem; font-weight: 500;">
                <strong>Published:</strong> {paper['published']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        selected = st.checkbox(
            f"Select for Q&A",
            key=f"paper_{index}",
            value=paper['id'] in st.session_state.selected_papers
        )
        
        if selected and paper['id'] not in st.session_state.selected_papers:
            st.session_state.selected_papers.append(paper['id'])
        elif not selected and paper['id'] in st.session_state.selected_papers:
            st.session_state.selected_papers.remove(paper['id'])
        
        
        with st.expander("Abstract", expanded=False):
            st.write(paper['abstract'])
        
        with st.expander("AI Summary", expanded=False):
            if 'summary' not in paper:
                with st.spinner("Generating summary..."):
                    paper['summary'] = summarizer.summarize_paper(paper['title'], paper['abstract'])
            st.write(paper['summary'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<a href="{paper["url"]}" target="_blank" class="link-button" style="text-decoration: none; color: #c4b5fd;">View Paper</a>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<a href="{paper["pdf_url"]}" target="_blank" class="link-button" style="text-decoration: none; color: #c4b5fd;">Download PDF</a>', unsafe_allow_html=True)
        
        st.divider()

def main():
    
    st.markdown("""
    <div class="main-header">
        <h1>🔬 AI Research Assistant</h1>
        <p>Search academic papers, get AI summaries, and ask questions about research findings</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    with st.sidebar:
        st.markdown("### 🎛️ Settings")
        
        max_results = st.slider(
            "Max Results",
            min_value=5,
            max_value=50,
            value=10,
            help="Number of papers to fetch"
        )
        
        st.markdown("### 📊 Statistics")
        if st.session_state.search_performed:
            st.markdown(f"""
            <div class="stats-container">
                <h4>📈 Search Stats</h4>
                <p><strong>Papers Found:</strong> {len(st.session_state.papers)}</p>
                <p><strong>Chat Messages:</strong> {len(st.session_state.chat_history)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 🔧 Tools")
        if st.button("🗑️ Clear All Data"):
            st.session_state.papers = []
            st.session_state.selected_papers = []
            st.session_state.chat_history = []
            st.session_state.search_performed = False
            st.rerun()
    
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="chat-container">
            <p style="margin: 0; color: #2c3e50; font-size: 16px; font-weight: 500;">
                Search for Research Papers!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        with st.form("search_form"):
            search_query = st.text_input(
                "Enter research keywords",
                placeholder="e.g., 'machine learning'"
            )
            
            col_search, col_example = st.columns([1, 1])
            with col_search:
                search_button = st.form_submit_button("Search Papers", use_container_width=True)
            with col_example:
                if st.form_submit_button("🎲 Try Example", use_container_width=True):
                    search_query = "artificial intelligence deep learning"
                    search_button = True
        
        
        if search_button and search_query:
            with st.spinner("🔍 Searching ArXiv database..."):
                papers = fetch_papers(search_query, max_results)
                
                if papers:
                    st.success(f"✅ Found {len(papers)} papers!")
                    
                    # Calculate relevance scores
                    with st.spinner("🧠 Calculating relevance scores..."):
                        papers = calculate_relevance_scores(search_query, papers)
                    
                    st.session_state.papers = papers
                    st.session_state.search_performed = True
                    
                else:
                    st.error("❌ No papers found. Try different keywords.")
        
        
        if st.session_state.papers:
            st.markdown("### 📚 Research Papers")
            
            
            col_filter, col_sort = st.columns(2)
            with col_filter:
                min_relevance = st.slider(
                    "Minimum Relevance Score",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=10
                )
            
            with col_sort:
                sort_by = st.selectbox(
                    "Sort by",
                    ["Relevance Score", "Publication Date", "Title"]
                )
            
            
            filtered_papers = [p for p in st.session_state.papers if p['relevance_score']*100 >= min_relevance]
            
            
            if sort_by == "Publication Date":
                filtered_papers.sort(key=lambda x: x['published'], reverse=True)
            elif sort_by == "Title":
                filtered_papers.sort(key=lambda x: x['title'])
            
            
            for i, paper in enumerate(filtered_papers):
                display_paper_card(paper, i)
    
    with col2:
        st.markdown("### 💭 Q&A Chat")
        
        
        st.markdown("""
        <div class="chat-container">
            <p style="margin: 0; color: #2c3e50; font-weight: 500;">
                Select papers from the left and ask questions about the research!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        
        if st.session_state.chat_history:
            st.markdown("#### 💭 Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:50]}...", expanded=i == len(st.session_state.chat_history)-1):
                    st.markdown(f"**Question:** {question}")
                    st.markdown(f"**Answer:** {answer}")
        
       
        with st.form("question_form"):
            question = st.text_area(
                "Ask a question about selected papers",
                placeholder="e.g., 'What are the key findings?', 'How do these approaches compare?'"
            )
            
            ask_button = st.form_submit_button("Ask Question", use_container_width=True)
        
        if ask_button and question:
            if not st.session_state.selected_papers:
                st.warning("⚠️ Please select at least one paper first!")
            else:
                with st.spinner("🧠 Analyzing papers and generating answer..."):
                    
                    papers_context = ""
                    for paper in st.session_state.papers:
                        if paper['id'] in st.session_state.selected_papers:
                            if 'summary' not in paper:
                                paper['summary'] = summarizer.summarize_paper(paper['title'], paper['abstract'])
                            
                            papers_context += f"""
                            Title: {paper['title']}
                            Authors: {', '.join(paper['authors'])}
                            Summary: {paper['summary']}
                            
                            """
                    
                    
                    answer = summarizer.answer_question(question, papers_context)
                    
                    
                    st.session_state.chat_history.append((question, answer))
                    
                    st.success("✅ Answer generated!")
                    st.rerun()
        
        
        if st.session_state.selected_papers:
            st.markdown("#### 📑 Selected Papers")
            for paper in st.session_state.papers:
                if paper['id'] in st.session_state.selected_papers:
                    st.markdown(f"• **{paper['title'][:50]}...**")

if __name__ == "__main__":
    main()
