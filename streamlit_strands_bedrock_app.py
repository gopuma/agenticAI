import streamlit as st
import boto3
import google.generativeai as genai
from strands import Agent
from strands.models import BedrockModel, CustomModel
from strands_tools import retrieve
import json
import asyncio
from datetime import datetime
import uuid
from typing import Dict, List, Any, Optional
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Advanced AI Agent with Bedrock & Gemini",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-response {
        background-color: #f0f2f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .memory-item {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title with styling
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">🤖 Advanced AI Agent Platform</h1>
    <p style="color: white; margin: 0;">Powered by Strands Agents, AWS Bedrock & Google Gemini</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "initialized" not in st.session_state:
        st.session_state.initialized = False

init_session_state()

# AWS Bedrock AgentCore with Memory, Runtime, Gateway
class BedrockAgentCore:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
        self.bedrock_agent_client = boto3.client('bedrock-agent-runtime', region_name=region_name)
        self.memory = {}
        self.session_id = st.session_state.session_id
        
    def store_memory(self, key: str, value: Any):
        """Store information in agent memory"""
        self.memory[key] = {
            "value": value,
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id
        }
        st.session_state.agent_memory = self.memory
        
    def get_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent memory"""
        return self.memory.get(key, {}).get("value")
    
    def get_all_memory(self) -> Dict:
        """Get all stored memories"""
        return self.memory
    
    async def invoke_knowledge_base(self, knowledge_base_id: str, query: str, max_results: int = 5):
        """Query Bedrock Knowledge Base"""
        try:
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': max_results
                    }
                }
            )
            return response.get('retrievalResults', [])
        except Exception as e:
            logger.error(f"Knowledge base query error: {e}")
            return []
    
    def invoke_model(self, model_id: str, prompt: str, max_tokens: int = 1000, temperature: float = 0.7):
        """Invoke Bedrock model through runtime"""
        try:
            if "anthropic.claude" in model_id:
                body = json.dumps({
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": max_tokens,
                    "temperature": temperature,
                    "stop_sequences": ["\n\nHuman:"]
                })
            elif "amazon.titan" in model_id:
                body = json.dumps({
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature
                    }
                })
            
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            
            if "anthropic.claude" in model_id:
                return response_body.get('completion', '')
            elif "amazon.titan" in model_id:
                return response_body.get('results', [{}])[0].get('outputText', '')
                
        except Exception as e:
            logger.error(f"Bedrock model invocation error: {e}")
            return f"Error: {str(e)}"

# Google Gemini Model Integration
class GeminiModel(CustomModel):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash", **kwargs):
        super().__init__(**kwargs)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
    
    async def generate(self, messages: List[Dict], **kwargs):
        """Generate response using Gemini API"""
        try:
            # Convert messages to Gemini format
            prompt = self._format_messages(messages)
            
            # Generate response
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get('temperature', 0.7),
                    max_output_tokens=kwargs.get('max_tokens', 1000)
                )
            )
            
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Gemini Error: {str(e)}"
    
    def _format_messages(self, messages: List[Dict]) -> str:
        """Format messages for Gemini"""
        formatted = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                formatted += f"User: {content}\n"
            else:
                formatted += f"Assistant: {content}\n"
        return formatted

# Advanced Agent Factory
class AdvancedAgentFactory:
    def __init__(self, bedrock_core: BedrockAgentCore):
        self.bedrock_core = bedrock_core
    
    def create_bedrock_agent(self, model_id: str, knowledge_base_id: str = None):
        """Create Strands Agent with Bedrock model and knowledge base"""
        model = BedrockModel(
            model_id=model_id,
            region=self.bedrock_core.region_name
        )
        
        tools = []
        if knowledge_base_id:
            knowledge_tool = retrieve(
                knowledge_base_id=knowledge_base_id,
                description="Search the knowledge base for relevant information",
                max_results=5
            )
            tools.append(knowledge_tool)
        
        agent = Agent(
            model=model,
            tools=tools,
            instructions="""
            You are an advanced AI assistant with access to specialized knowledge.
            Use the knowledge base to provide accurate, contextual responses.
            Store important conversation context in memory for future reference.
            Be helpful, accurate, and provide detailed explanations.
            """
        )
        return agent
    
    def create_gemini_agent(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Create Strands Agent with Gemini model"""
        model = GeminiModel(api_key=api_key, model_name=model_name)
        
        agent = Agent(
            model=model,
            instructions="""
            You are an AI assistant powered by Google Gemini.
            Provide helpful, accurate, and detailed responses.
            Be conversational and engaging while maintaining professionalism.
            """
        )
        return agent

# Sidebar Configuration
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Model Selection
    model_provider = st.selectbox(
        "Select Model Provider",
        ["AWS Bedrock", "Google Gemini", "Multi-Agent (Both)"]
    )
    
    st.subheader("AWS Configuration")
    aws_region = st.selectbox("AWS Region", ["us-east-1", "us-west-2", "eu-west-1"])
    
    if model_provider in ["AWS Bedrock", "Multi-Agent (Both)"]:
        bedrock_model = st.selectbox(
            "Bedrock Model",
            [
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "amazon.titan-text-express-v1"
            ]
        )
        
        knowledge_base_id = st.text_input(
            "Bedrock Knowledge Base ID",
            help="Enter your Bedrock Knowledge Base ID",
            placeholder="XXXXXXXXXX"
        )
    
    if model_provider in ["Google Gemini", "Multi-Agent (Both)"]:
        st.subheader("Google Gemini Configuration")
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Enter your Google AI Studio API key"
        )
        
        gemini_model = st.selectbox(
            "Gemini Model",
            ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro"]
        )
    
    # Agent Parameters
    st.subheader("Agent Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    max_tokens = st.slider("Max Tokens", 100, 4000, 1000, 100)
    
    # Memory Management
    st.subheader("Memory Management")
    if st.button("Clear Agent Memory"):
        st.session_state.agent_memory = {}
        st.success("Memory cleared!")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.success("Chat history cleared!")

# Initialize Bedrock AgentCore
bedrock_core = BedrockAgentCore(region_name=aws_region)
agent_factory = AdvancedAgentFactory(bedrock_core)

# Main Content Area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("metadata"):
                with st.expander("Additional Info"):
                    st.json(message["metadata"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response based on selected provider
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if model_provider == "AWS Bedrock":
                        # Use Bedrock agent
                        agent = agent_factory.create_bedrock_agent(
                            model_id=bedrock_model,
                            knowledge_base_id=knowledge_base_id if knowledge_base_id else None
                        )
                        response = agent(prompt)
                        
                        # Store in memory
                        bedrock_core.store_memory(f"conversation_{len(st.session_state.messages)}", {
                            "user_input": prompt,
                            "agent_response": response,
                            "model": bedrock_model
                        })
                        
                    elif model_provider == "Google Gemini":
                        if not gemini_api_key:
                            response = "Please provide your Gemini API key in the sidebar."
                        else:
                            # Use Gemini agent
                            agent = agent_factory.create_gemini_agent(gemini_api_key, gemini_model)
                            response = await agent.generate([{"role": "user", "content": prompt}])
                    
                    elif model_provider == "Multi-Agent (Both)":
                        if not gemini_api_key:
                            response = "Please provide your Gemini API key for multi-agent mode."
                        else:
                            # Use both agents and compare/combine responses
                            bedrock_agent = agent_factory.create_bedrock_agent(
                                model_id=bedrock_model,
                                knowledge_base_id=knowledge_base_id if knowledge_base_id else None
                            )
                            gemini_agent = agent_factory.create_gemini_agent(gemini_api_key, gemini_model)
                            
                            bedrock_response = bedrock_agent(prompt)
                            gemini_response = await gemini_agent.generate([{"role": "user", "content": prompt}])
                            
                            response = f"""
**Bedrock Response ({bedrock_model}):**
{bedrock_response}

---

**Gemini Response ({gemini_model}):**
{gemini_response}
                            """
                    
                    # Display response
                    st.markdown(response)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "metadata": {
                            "provider": model_provider,
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

with col2:
    st.subheader("🧠 Agent Memory")
    
    # Display current memory
    if st.session_state.agent_memory:
        for key, value in st.session_state.agent_memory.items():
            with st.expander(f"Memory: {key}"):
                st.json(value)
    else:
        st.info("No memories stored yet. Start chatting to build memory!")
    
    # Knowledge Base Query Section
    if knowledge_base_id:
        st.subheader("📚 Knowledge Base")
        kb_query = st.text_input("Query Knowledge Base:")
        if st.button("Search KB") and kb_query:
            with st.spinner("Searching knowledge base..."):
                results = asyncio.run(bedrock_core.invoke_knowledge_base(knowledge_base_id, kb_query))
                if results:
                    for i, result in enumerate(results[:3]):
                        with st.expander(f"Result {i+1}"):
                            st.write(result.get('content', {}).get('text', 'No content'))
                            if result.get('metadata'):
                                st.json(result['metadata'])
                else:
                    st.info("No results found in knowledge base.")
    
    # System Status
    st.subheader("📊 System Status")
    st.metric("Messages", len(st.session_state.messages))
    st.metric("Memory Items", len(st.session_state.agent_memory))
    st.metric("Session ID", st.session_state.session_id[:8] + "...")

# Footer
st.markdown("---")
st.markdown("""
**Features:**
- 🤖 Strands Agents SDK integration
- ☁️ AWS Bedrock AgentCore with Memory, Runtime & Gateway  
- 📚 Bedrock Knowledge Base integration
- 🔍 Google Gemini LLM API support
- 💾 Persistent memory across conversations
- 🔄 Multi-agent comparison mode
""")

# Instructions
with st.expander("📋 Setup Instructions"):
    st.markdown("""
    ## Setup Steps:
    
    1. **Install Dependencies:**
       ```bash
       pip install -r requirements.txt
       ```
    
    2. **Configure AWS:**
       ```bash
       aws configure
       # Ensure you have Bedrock permissions
       ```
    
    3. **Create Knowledge Base:**
       - Set up Bedrock Knowledge Base in AWS Console
       - Note the Knowledge Base ID
    
    4. **Get Gemini API Key:**
       - Visit Google AI Studio
       - Generate API key
       - Enter in sidebar
    
    5. **Run Application:**
       ```bash
       streamlit run app.py
       ```
    """)
