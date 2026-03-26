"""
NOVA AI Platform — Demo Script (Task 5)
Demonstrates 3 end-to-end scenarios through the full multi-agent pipeline.

Scenario 1: Happy path — order status (auto-resolved via MCP)
Scenario 2: Product knowledge — ingredient query (answered via RAG)
Scenario 3: Escalation — angry customer with legal threat (human handoff)

Usage:
    python task5_demo.py
    python task5_demo.py --scenario 1   # Run specific scenario
    python task5_demo.py --all          # Run all scenarios
"""

import os
import sys
import json
import argparse
from datetime import datetime

sys.path.insert(0, ".")

from task5_nova_platform import NOVAPlatform, NOVAPlatformConfig


def get_api_key() -> str:
    """Get OpenRouter API key from environment or prompt."""
    key = os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        print("⚠️  OPENROUTER_API_KEY not set in environment.")
        print("    Set it with: export OPENROUTER_API_KEY=your_key")
        print("    Or add to .env file\n")
    return key


def print_scenario_header(num: int, title: str, description: str):
    print(f"\n{'#'*65}")
    print(f"# SCENARIO {num}: {title}")
    print(f"# {description}")
    print(f"{'#'*65}")


def scenario_1_order_status(platform: NOVAPlatform):
    """
    Scenario 1: Happy Path — Order Status Query
    Expected flow: TicketRouter → SupportAgent (get_order_status) → BrandVoiceAgent → AuditLogger
    """
    print_scenario_header(
        1,
        "Order Status — Happy Path",
        "Customer asks about a late order → MCP lookup → brand voice response"
    )

    result = platform.process_ticket(
        ticket="Hi! My order ORD-1042 was placed a week ago but I haven't received any updates. "
               "Can you tell me where it is? My tracking number isn't working.",
        customer_id="CUST-1010",
        session_id="demo-scenario-1"
    )

    print(f"\n📊 Scenario 1 Results:")
    print(f"  Intent:    {result['intent']}")
    print(f"  Escalated: {result['escalated']}")
    print(f"  Tools:     {result['tool_calls']} MCP calls")
    print(f"  Audit:     {result['audit_trail_length']} trail entries")


def scenario_2_rag_query(platform: NOVAPlatform):
    """
    Scenario 2: Product Knowledge — RAG Pipeline
    Expected flow: TicketRouter → RAGAgent (hybrid search + rerank) → BrandVoiceAgent → AuditLogger
    """
    print_scenario_header(
        2,
        "Product Knowledge — RAG Pipeline",
        "Customer asks ingredient question → RAG retrieval → grounded answer"
    )

    result = platform.process_ticket(
        ticket="I'm pregnant and want to know which NOVA skincare products are safe to use. "
               "Specifically, does the Glow Serum contain retinol? And what about the Vitamin C serum — "
               "is it safe during pregnancy?",
        customer_id="CUST-1005",
        session_id="demo-scenario-2"
    )

    print(f"\n📊 Scenario 2 Results:")
    print(f"  Intent:    {result['intent']}")
    print(f"  Escalated: {result['escalated']}")
    print(f"  Audit:     {result['audit_trail_length']} trail entries")


def scenario_3_escalation(platform: NOVAPlatform):
    """
    Scenario 3: Escalation — Angry Customer with Legal Threat
    Expected flow: TicketRouter → EscalationAgent (immediate escalate) → AuditLogger
    """
    print_scenario_header(
        3,
        "Escalation — Human-in-the-Loop",
        "Furious customer with legal threat → immediate escalation to human agent"
    )

    result = platform.process_ticket(
        ticket="THIS IS ABSOLUTELY OUTRAGEOUS!!! I've contacted your useless support team "
               "FOUR TIMES about my missing order and nobody has helped me. My solicitor "
               "is already aware of this situation and I will be going to Trading Standards "
               "if this is not resolved TODAY. I'm also posting about this on every social "
               "media platform I can find. Sort it OUT!!!",
        customer_id="CUST-1020",
        session_id="demo-scenario-3"
    )

    print(f"\n📊 Scenario 3 Results:")
    print(f"  Intent:    {result['intent']}")
    print(f"  Escalated: {'✅ YES — Human agent notified' if result['escalated'] else '❌ NOT escalated (unexpected)'}")
    print(f"  Audit:     {result['audit_trail_length']} trail entries")
    print(f"\n  ℹ️  In production, this would:")
    print(f"     1. Send alert to senior support agent dashboard")
    print(f"     2. Attach full context summary + order history")
    print(f"     3. Trigger 15-minute SLA response timer")


def run_all_scenarios(platform: NOVAPlatform):
    """Run all three demo scenarios."""
    print("\n" + "="*65)
    print("  NOVA AI PLATFORM — TASK 5 DEMO")
    print("  3 End-to-End Multi-Agent Scenarios")
    print("="*65)

    scenario_1_order_status(platform)
    scenario_2_rag_query(platform)
    scenario_3_escalation(platform)

    # Summary stats
    print("\n" + "="*65)
    print("  DEMO COMPLETE — Summary")
    print("="*65)

    # Load and display traces
    try:
        with open("nova_traces.json") as f:
            lines = f.readlines()
        traces = [json.loads(l) for l in lines[-3:]]  # Last 3 sessions

        print(f"\n  Sessions logged: {len(traces)}")
        for trace in traces:
            print(f"\n  Session: {trace['session_id']}")
            print(f"    Intent:   {trace.get('intent')}")
            print(f"    Escalated: {trace.get('escalated')}")
            print(f"    Tools:    {len(trace.get('tool_calls', []))} calls")
            print(f"    Audit:    {len(trace.get('audit_trail', []))} entries")
    except FileNotFoundError:
        print("  (nova_traces.json not found — run demo first)")


def build_graph_visualization(platform: NOVAPlatform):
    """Generate and save the LangGraph visualization."""
    print("\nGenerating LangGraph visualization...")
    try:
        platform.visualize_graph("nova_agent_graph.png")
        print("✅ Graph saved to nova_agent_graph.png")
    except Exception as e:
        print(f"Could not generate PNG ({e})")
        # Print mermaid as fallback
        print("\nMermaid diagram (copy to https://mermaid.live):")
        print("""
graph TD
    START --> TicketRouter
    TicketRouter -- order_status/return --> SupportAgent
    TicketRouter -- product_query/sizing --> RAGAgent
    TicketRouter -- recommendation --> PersonalizationAgent
    TicketRouter -- escalate/injection --> EscalationAgent
    SupportAgent --> BrandVoiceAgent
    RAGAgent --> BrandVoiceAgent
    PersonalizationAgent --> BrandVoiceAgent
    EscalationAgent --> AuditLogger
    BrandVoiceAgent --> AuditLogger
    AuditLogger --> END
""")
        # Save mermaid diagram
        with open("nova_agent_graph.png", "w") as f:
            f.write("# Mermaid Diagram — render at https://mermaid.live\n")
            f.write("""graph TD
    START --> TicketRouter
    TicketRouter -- order_status/return --> SupportAgent
    TicketRouter -- product_query/sizing --> RAGAgent
    TicketRouter -- recommendation --> PersonalizationAgent
    TicketRouter -- escalate/injection --> EscalationAgent
    SupportAgent --> BrandVoiceAgent
    RAGAgent --> BrandVoiceAgent
    PersonalizationAgent --> BrandVoiceAgent
    EscalationAgent --> AuditLogger
    BrandVoiceAgent --> AuditLogger
    AuditLogger --> END""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOVA AI Platform Demo")
    parser.add_argument("--scenario", type=int, choices=[1, 2, 3],
                        help="Run a specific scenario")
    parser.add_argument("--all", action="store_true", help="Run all 3 scenarios")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG (faster startup)")
    parser.add_argument("--visualize", action="store_true", help="Generate graph visualization only")
    args = parser.parse_args()

    # Initialize platform
    config = NOVAPlatformConfig(
        openrouter_api_key=get_api_key(),
        llm_model="mistralai/mistral-7b-instruct:free",
        mock_db_path="nova_mock_db.json",
        chroma_path="./chroma_db",
        audit_log_path="nova_traces.json",
        use_rag=not args.no_rag,
        use_brand_voice=True
    )

    platform = NOVAPlatform(config)

    if args.visualize:
        build_graph_visualization(platform)
    elif args.scenario == 1:
        scenario_1_order_status(platform)
    elif args.scenario == 2:
        scenario_2_rag_query(platform)
    elif args.scenario == 3:
        scenario_3_escalation(platform)
    else:
        # Default: run all scenarios
        run_all_scenarios(platform)
        build_graph_visualization(platform)
