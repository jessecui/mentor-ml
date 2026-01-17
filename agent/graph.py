"""
LangGraph agent for MentorML.

A ReAct-style agent that:
1. Receives user queries about AI/ML concepts
2. Can retrieve relevant diagrams using SigLIP
3. Generates teaching responses with visual aids
4. Maintains conversation state via Redis checkpointing
"""

import os
from typing import TYPE_CHECKING

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from agent.tools import create_retrieval_tool

if TYPE_CHECKING:
    from model.scorer import SigLIPScorer

# Module-level agent (singleton)
_agent = None

# System prompt for the teaching agent
SYSTEM_PROMPT = """You are MentorML, an expert AI/ML teaching assistant.

Your purpose is to help users understand machine learning concepts through clear explanations
and visual diagrams. You have access to a corpus of diagrams from Jay Alammar's famous
"Illustrated" blog posts covering Transformers, BERT, GPT-2, Word2vec, and Stable Diffusion.

Guidelines:
1. When explaining a concept, use the retrieve_diagram tool to find a relevant visualization
2. Reference the diagram in your explanation (e.g., "As shown in the diagram...")
3. Break down complex topics into digestible pieces
4. Use analogies and examples when helpful
5. If the user asks a follow-up question, consider whether a new diagram would help

When you retrieve a diagram, include its ID in your response so the user can view it.
Format: [diagram: diagram_XXX]

Be friendly, encouraging, and patient. Your goal is to make AI/ML concepts accessible."""


def create_agent(scorer: "SigLIPScorer", checkpointer: BaseCheckpointSaver | None = None):
    """
    Create a LangGraph ReAct agent with diagram retrieval.
    
    Args:
        scorer: Pre-initialized SigLIPScorer instance
        checkpointer: Optional checkpoint saver for conversation state.
                     If None, uses InMemorySaver (state lost on restart).
        
    Returns:
        Compiled LangGraph agent
    """
    global _agent
    
    # Use in-memory saver if no checkpointer provided
    if checkpointer is None:
        checkpointer = InMemorySaver()
    
    # Create the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
    )
    
    # Create retrieval tool bound to the scorer
    retrieval_tool = create_retrieval_tool(scorer)
    
    # Create ReAct agent with tools
    _agent = create_react_agent(
        model=llm,
        tools=[retrieval_tool],
        checkpointer=checkpointer,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    
    return _agent


def get_agent():
    """Get the initialized agent. Raises if not created."""
    if _agent is None:
        raise RuntimeError("Agent not initialized. Call create_agent() first.")
    return _agent
