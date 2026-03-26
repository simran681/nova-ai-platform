"""
NOVA AI Platform — MCP Demo (Task 2)
Demonstrates the compound support scenario using NOVA's 5 backend tools.

Compound Scenario:
  A customer (Sarah) contacts support with a complex multi-part enquiry:
  1. Her recent order hasn't arrived
  2. She wants to return a product from a previous order
  3. She wants a recommendation for an oily-skin alternative

Usage:
    python task2_mcp/demo.py
    python task2_mcp/demo.py --server  # Use HTTP server instead of local client
"""

import json
import sys
import argparse
from datetime import datetime

# Use local client (no server needed) by default
sys.path.insert(0, ".")
from task2_mcp.client import NOVAMCPClientLocal, NOVAMCPClient


def print_section(title: str):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def print_result(tool: str, result: dict, indent: int = 2):
    prefix = " " * indent
    print(f"{prefix}[{tool}]")
    for key, value in result.items():
        if isinstance(value, list) and len(value) > 3:
            print(f"{prefix}  {key}: [{len(value)} items] {value[:2]}...")
        elif isinstance(value, dict):
            print(f"{prefix}  {key}: {{...}}")
        else:
            print(f"{prefix}  {key}: {value}")


def run_compound_scenario(client):
    """
    Compound Demo Scenario:
    Sarah (CUST-1000) contacts NOVA support with three issues:
    - Her Glow Serum order (ORD-1000) hasn't arrived
    - She wants to return the Moisturizer from order ORD-1001
    - She wants a recommendation for something suitable for oily skin
    """

    print_section("NOVA MCP Demo — Compound Support Scenario")
    print("""
SCENARIO: Sarah contacts NOVA support.

Her message:
  "Hi! I ordered the Glow Serum last week (order ORD-1000) and it still
   hasn't arrived. Also, the moisturizer from my last order (ORD-1001)
   isn't working for my oily skin — I'd like to return it. And can you
   suggest something better for oily skin? Thanks!"

This triggers 4 sequential MCP tool calls:
  1. get_order_status(ORD-1000)
  2. get_customer_history(CUST-1000)
  3. process_return(ORD-1001, "not suitable for oily skin")
  4. recommend_products(CUST-1000, "oily skin alternative to moisturizer")
""")

    # ── Step 1: Check order status ────────────────────────────────────────────
    print_section("Step 1: Check Order Status")
    print("  → Calling: get_order_status('ORD-1000')")

    order_result = client.get_order_status("ORD-1000")
    print(f"  Status:     {order_result.get('status', 'ERROR')}")
    print(f"  Est. Del.:  {order_result.get('estimated_delivery', 'N/A')}")
    print(f"  Tracking:   {order_result.get('tracking_number', 'N/A')}")
    print(f"  Tracking URL: {order_result.get('tracking_url', 'N/A')}")

    if order_result.get("items"):
        print(f"  Items:      {[i['name'] for i in order_result['items']]}")

    # ── Step 2: Fetch customer profile ────────────────────────────────────────
    print_section("Step 2: Fetch Customer Profile")
    print("  → Calling: get_customer_history('CUST-1000')")

    customer_result = client.get_customer_history("CUST-1000")
    print(f"  Customer:   {customer_result.get('name', 'N/A')}")
    print(f"  Skin Type:  {customer_result.get('skin_type', 'N/A')}")
    print(f"  Loyalty:    {customer_result.get('loyalty_tier', 'N/A').upper()} "
          f"({customer_result.get('loyalty_points', 0)} pts)")
    print(f"  VIP:        {customer_result.get('is_vip', False)}")
    print(f"  Categories: {customer_result.get('preferred_categories', [])}")

    # ── Step 3: Process return ────────────────────────────────────────────────
    print_section("Step 3: Process Return")
    print("  → Calling: process_return('ORD-1001', reason='not suitable for oily skin')")

    return_result = client.process_return(
        order_id="ORD-1001",
        reason="Product not suitable for oily skin type",
        items=None  # Return all items in order
    )
    if return_result.get("success"):
        print(f"  ✅ Return Approved")
        print(f"  Return ID:    {return_result.get('return_id', 'N/A')}")
        print(f"  Return Label: {return_result.get('return_label', 'N/A')}")
        print(f"  Refund:       ${return_result.get('refund_amount', 0):.2f}")
        print(f"  Est. Days:    {return_result.get('estimated_refund_days', 5)} business days")
    else:
        print(f"  ⚠️ Return Not Possible: {return_result.get('reason', return_result.get('error'))}")

    # ── Step 4: Recommend products ────────────────────────────────────────────
    print_section("Step 4: Personalised Recommendations")
    print("  → Calling: recommend_products('CUST-1000', context='oily skin, skincare')")

    recs_result = client.recommend_products(
        customer_id="CUST-1000",
        context="oily skin, looking for a lighter moisturizer or serum",
        category="skincare"
    )

    print(f"  Top {len(recs_result.get('recommendations', []))} Recommendations:")
    for i, rec in enumerate(recs_result.get("recommendations", [])[:5], 1):
        print(f"\n  {i}. {rec['name']} — ${rec['price']}")
        print(f"     Rating: {rec.get('rating', 'N/A')}/5 | "
              f"Bestseller: {'✅' if rec.get('is_bestseller') else '❌'}")
        print(f"     Why: {rec.get('recommendation_reason', '')}")

    # ── Step 5: Compose final response ───────────────────────────────────────
    print_section("Final Customer Response (Brand Voice)")
    customer_name = customer_result.get("name", "there").split()[0]
    order_status = order_result.get("status", "in transit")
    est_delivery = order_result.get("estimated_delivery", "soon")
    refund_amt = return_result.get("refund_amount", 0)
    top_rec = recs_result.get("recommendations", [{}])[0]

    response = f"""
Hi {customer_name}! 💫

We're on it! Here's a full update for you:

📦 ORDER ORD-1000 — STATUS: {order_status.upper()}
Your Glow Serum is on its way and estimated to arrive by {est_delivery}.
Track it here: {order_result.get('tracking_url', 'https://track.nova.com')}

↩️ RETURN FOR ORD-1001
{"We've approved your return! Your return label is ready at: " + return_result.get('return_label', '') if return_result.get('success') else "We're looking into your return — our team will follow up shortly."}
{"Your refund of $" + str(refund_amt) + " will reach you within 5 business days once we receive the item." if return_result.get('success') else ""}

✨ RECOMMENDATION FOR OILY SKIN
Based on your skin type, we think you'll love:
→ {top_rec.get('name', 'our bestselling oily-skin serum')} — ${top_rec.get('price', '')}
   {top_rec.get('recommendation_reason', '')}

Is there anything else I can help you with today? We're always here for you! 🌟
— Your NOVA Team
"""
    print(response)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("Audit Summary")
    print(f"  Tools Called:     4")
    print(f"  Session ID:       {client.session_id}")
    print(f"  Audit Log:        audit_log.jsonl (appended)")
    print(f"  Timestamp:        {datetime.utcnow().isoformat()}Z")


def run_unit_tests(client):
    """Quick unit tests for all 5 tools."""
    print_section("Unit Tests — All 5 MCP Tools")

    tests = [
        ("get_order_status", lambda: client.get_order_status("ORD-1000")),
        ("get_order_status (invalid)", lambda: client.get_order_status("ORD-9999")),
        ("process_return", lambda: client.process_return("ORD-1000", "wrong size")),
        ("get_product_info", lambda: client.get_product_info("PROD-1000")),
        ("get_product_info (name search)", lambda: client.get_product_info("Glow Serum")),
        ("get_customer_history", lambda: client.get_customer_history("CUST-1000")),
        ("recommend_products", lambda: client.recommend_products("CUST-1000", "gift ideas")),
    ]

    passed = 0
    for name, fn in tests:
        try:
            result = fn()
            has_error = "error" in result
            status = "⚠️ API ERROR" if has_error else "✅ PASS"
            if not has_error:
                passed += 1
            print(f"  {status}  {name}")
            if has_error:
                print(f"          Error: {result['error']}")
        except Exception as e:
            print(f"  ❌ FAIL  {name}: {e}")

    print(f"\n  Results: {passed}/{len(tests)} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NOVA MCP Demo")
    parser.add_argument("--server", action="store_true",
                        help="Use HTTP server (requires server.py running on :8001)")
    parser.add_argument("--unit-tests", action="store_true",
                        help="Run unit tests instead of compound demo")
    args = parser.parse_args()

    if args.server:
        print("Using HTTP MCP Server at http://localhost:8001")
        client = NOVAMCPClient(base_url="http://localhost:8001")
        health = client.health_check()
        if health.get("status") != "ok":
            print(f"⚠️ Server not responding: {health}")
            sys.exit(1)
    else:
        print("Using Local MCP Client (reading nova_mock_db.json directly)")
        client = NOVAMCPClientLocal(
            db_path="nova_mock_db.json",
            audit_log_path="audit_log.jsonl"
        )

    if args.unit_tests:
        run_unit_tests(client)
    else:
        run_compound_scenario(client)
        print("\n  Running quick unit tests...")
        run_unit_tests(client)
