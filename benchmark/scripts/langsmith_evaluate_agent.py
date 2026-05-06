"""
End-to-end agent eval against the LangSmith retrieval dataset.

Target: invokes the full LangGraph agent (planner -> executor -> MCP-backed
retrieve_diagram) on each query and returns the agent's response plus the
diagram_ids it actually retrieved.

Evaluators:
- tool_call_correctness  (well-grounded): is the dataset's relevant_image_id
                          in the agent's retrieval set? (Agent may retrieve up
                          to 2 per response.)
- explanation_quality    (LLM judge, ungrounded — no reference answers):
                          accuracy + accessibility, rated 1-5 by Gemini Flash.

Usage:
    export LANGSMITH_API_KEY=...
    export GOOGLE_API_KEY=...
    python benchmark/scripts/langsmith_evaluate_agent.py --limit 3
    python benchmark/scripts/langsmith_evaluate_agent.py --no-vision --no-judge
    python benchmark/scripts/langsmith_evaluate_agent.py
"""

import argparse
import asyncio
import json
import os
import sys
import uuid
from contextlib import AsyncExitStack
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET_NAME = "mentorml-diagram-retrieval"


# =============================================================================
# AGENT TARGET
# =============================================================================

def make_agent_target(agent):
    """Build an async target function that invokes the agent end-to-end."""
    from langchain_core.messages import HumanMessage

    async def target(inputs: dict) -> dict:
        thread_id = f"eval-{uuid.uuid4()}"
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=inputs["query"])]},
            config={"configurable": {"thread_id": thread_id}},
        )

        msgs = result.get("messages", [])

        # Final AI text response (skip tool-only AI messages)
        response_text = ""
        for msg in reversed(msgs):
            if msg.type != "ai":
                continue
            if getattr(msg, "tool_calls", None) and not msg.content:
                continue
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            if isinstance(content, str) and content.strip():
                response_text = content
                break

        # Diagram IDs the agent actually retrieved (in call order, deduped)
        retrieved: list[str] = []
        for msg in msgs:
            if msg.type == "tool" and msg.name == "retrieve_diagram":
                try:
                    tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
                    diagram_id = (tool_result or {}).get("id")
                    if diagram_id and diagram_id not in retrieved:
                        retrieved.append(diagram_id)
                except (json.JSONDecodeError, TypeError):
                    pass

        plan = result.get("plan")
        plan_topic = plan.topic if plan else None

        return {
            "response": response_text,
            "retrieved_diagram_ids": retrieved,
            "plan_topic": plan_topic,
        }

    return target


# =============================================================================
# EVALUATOR A: tool-call correctness
# =============================================================================

def tool_call_correctness(run, example) -> dict:
    """1 if the dataset's relevant_image_id appears in the agent's retrieval set."""
    expected = (example.outputs or {}).get("relevant_image_id")
    retrieved = (run.outputs or {}).get("retrieved_diagram_ids", []) or []
    return {
        "key": "tool_call_correctness",
        "score": 1 if expected in retrieved else 0,
        "comment": f"expected={expected} retrieved={retrieved}",
    }


# =============================================================================
# EVALUATOR B: LLM judge for explanation quality
# =============================================================================

JUDGE_PROMPT = """You are evaluating an AI/ML teaching response.

Question asked: {query}

Response:
{response}

Rate the response on two dimensions, each 1-5 (5 = excellent, 1 = poor):
- accuracy: is the technical ML content factually correct?
- accessibility: is it clear and well-paced for a learner with some ML background?

Respond with ONLY a JSON object on a single line, no other text, no markdown:
{{"accuracy": <int 1-5>, "accessibility": <int 1-5>, "reasoning": "<one short sentence>"}}"""


def make_quality_judge():
    """Build an async LLM-judge evaluator. Returns a function suitable for evaluators=[...]."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.0, thinking_level="low")

    async def explanation_quality(run, example) -> list[dict]:
        response = (run.outputs or {}).get("response", "")
        query = (example.inputs or {}).get("query", "")

        if not response:
            return [
                {"key": "accuracy", "score": 0, "comment": "no response"},
                {"key": "accessibility", "score": 0, "comment": "no response"},
            ]

        judgment = await llm.ainvoke([
            SystemMessage(content="You are a strict but fair AI/ML educator."),
            HumanMessage(content=JUDGE_PROMPT.format(query=query, response=response)),
        ])

        text = getattr(judgment, "text", None) or judgment.content
        if isinstance(text, list):
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        text = (text or "").strip()

        # Strip markdown fencing if the model added it despite instructions
        if "```" in text:
            chunks = text.split("```")
            if len(chunks) >= 2:
                inner = chunks[1]
                if inner.startswith("json"):
                    inner = inner[4:]
                text = inner.strip()

        try:
            parsed = json.loads(text)
            acc = int(parsed.get("accuracy", 0))
            access = int(parsed.get("accessibility", 0))
            reasoning = str(parsed.get("reasoning", ""))[:200]
            return [
                {"key": "accuracy", "score": acc / 5.0, "comment": reasoning},
                {"key": "accessibility", "score": access / 5.0},
            ]
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            return [
                {"key": "accuracy", "score": 0, "comment": f"judge parse error: {text[:120]!r}"},
                {"key": "accessibility", "score": 0},
            ]

    return explanation_quality


# =============================================================================
# MAIN
# =============================================================================

async def amain():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--limit", "-n", type=int, default=None)
    parser.add_argument("--no-vision", action="store_true",
                        help="Disable Gemini vision review on retrieved diagrams (~50%% faster).")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip the LLM-judge evaluator (only run tool_call_correctness).")
    parser.add_argument("--max-concurrency", type=int, default=2,
                        help="Parallel agent invocations. Higher = faster but risks Gemini rate limits.")
    args = parser.parse_args()

    # Vision review is read by the MCP subprocess at spawn time — must set BEFORE
    # building the MCP client.
    if args.no_vision:
        os.environ["ENABLE_VISION"] = "false"

    from langgraph.checkpoint.memory import InMemorySaver
    from langsmith import Client, aevaluate

    from agent.graph import create_agent
    from agent.tools import load_persistent_mcp_tools

    client = Client()

    # Resolve dataset
    if args.limit is not None:
        all_examples = list(client.list_examples(dataset_name=args.dataset_name))
        data = all_examples[:args.limit]
        print(f"Running against {len(data)}/{len(all_examples)} examples (--limit {args.limit})")
    else:
        existing = list(client.list_datasets(dataset_name=args.dataset_name))
        if not existing:
            print(f"❌ Dataset '{args.dataset_name}' not found. Run upload_to_langsmith.py first.")
            sys.exit(1)
        data = args.dataset_name
        print(f"Running against full dataset '{args.dataset_name}'")

    # Build evaluator list
    evaluators: list = [tool_call_correctness]
    if not args.no_judge:
        evaluators.append(make_quality_judge())
    print(f"Evaluators: {[getattr(e, '__name__', repr(e)) for e in evaluators]}")
    print(f"Vision review: {'disabled' if args.no_vision else 'enabled'}")
    print(f"Max concurrency: {args.max_concurrency}")
    print()

    async with AsyncExitStack() as stack:
        print("🔌 Spawning diagram MCP server (stdio, persistent session)...")
        tools = await load_persistent_mcp_tools(stack)
        print(f"   ✅ Loaded {len(tools)} tool(s) from MCP server")

        agent = create_agent(tools, checkpointer=InMemorySaver())
        target = make_agent_target(agent)

        print("\n" + "=" * 60)
        print("Evaluating full LangGraph agent end-to-end")
        print("=" * 60)

        results = await aevaluate(
            target,
            data=data,
            evaluators=evaluators,
            experiment_prefix="agent-e2e",
            max_concurrency=args.max_concurrency,
        )
        print(f"\n✅ Agent eval complete: {results.experiment_name}")
        print("View results in the LangSmith UI under your dataset.")


def main():
    asyncio.run(amain())


if __name__ == "__main__":
    main()
