"""
NOVA AI Platform — MCP Server (Task 2)
Provides 5 backend tools via FastAPI with full audit logging.

Tools:
  1. get_order_status(order_id)
  2. process_return(order_id, reason, items)
  3. get_product_info(product_id)
  4. get_customer_history(customer_id)
  5. recommend_products(customer_id, context)

Run: uvicorn task2_mcp.server:app --reload --port 8001
"""

import json
import time
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Load mock database ────────────────────────────────────────────────────────
DB_PATH = Path(__file__).parent.parent / "nova_mock_db.json"
AUDIT_LOG_PATH = Path(__file__).parent.parent / "audit_log.jsonl"


def load_db() -> dict:
    """Load the NOVA mock database."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"nova_mock_db.json not found. Run generate_mock_db.py first."
        )
    with open(DB_PATH) as f:
        return json.load(f)


def log_audit(session_id: str, tool: str, params: dict, response: dict, latency_ms: float):
    """Append an audit entry to audit_log.jsonl."""
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "tool": tool,
        "params": params,
        "response_summary": str(response)[:200],  # Truncate for log readability
        "success": "error" not in response,
        "latency_ms": round(latency_ms, 2)
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="NOVA MCP Server",
    description="NOVA AI Platform — Backend Tool Integration (Task 2)",
    version="1.0.0"
)

DB = {}  # Loaded on startup


@app.on_event("startup")
async def startup():
    global DB
    DB = load_db()
    # Build lookup indexes for performance
    DB["_order_index"] = {o["order_id"]: o for o in DB["orders"]}
    DB["_customer_index"] = {c["customer_id"]: c for c in DB["customers"]}
    DB["_product_index"] = {p["product_id"]: p for p in DB["products"]}
    print(f"NOVA MCP Server started. Loaded {len(DB['orders'])} orders, "
          f"{len(DB['customers'])} customers, {len(DB['products'])} products.")


# ── Request / Response Models ──────────────────────────────────────────────────
class OrderStatusRequest(BaseModel):
    order_id: str
    session_id: str = ""


class ProcessReturnRequest(BaseModel):
    order_id: str
    reason: str
    items: Optional[List[str]] = None  # List of product_ids to return (None = all)
    session_id: str = ""


class ProductInfoRequest(BaseModel):
    product_id: str
    session_id: str = ""


class CustomerHistoryRequest(BaseModel):
    customer_id: str
    session_id: str = ""


class RecommendRequest(BaseModel):
    customer_id: str
    context: str = ""  # e.g., "oily skin", "gift for sister", "winter wardrobe"
    category: Optional[str] = None
    session_id: str = ""


# ── Tool 1: Get Order Status ──────────────────────────────────────────────────
@app.post("/tools/get_order_status")
async def get_order_status(req: OrderStatusRequest):
    """
    Retrieve real-time order status, tracking, and delivery estimate.
    MCP Tool 1 of 5.
    """
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    order = DB["_order_index"].get(req.order_id)

    if not order:
        result = {"error": f"Order {req.order_id} not found.", "order_id": req.order_id}
        log_audit(session_id, "get_order_status", {"order_id": req.order_id},
                  result, (time.perf_counter() - t0) * 1000)
        raise HTTPException(status_code=404, detail=result["error"])

    result = {
        "order_id": order["order_id"],
        "status": order["status"],
        "status_description": _status_description(order["status"]),
        "items": [
            {"name": item["product_name"], "qty": item["quantity"]}
            for item in order["items"]
        ],
        "total_amount": order["total_amount"],
        "order_date": order["order_date"],
        "estimated_delivery": order["estimated_delivery"],
        "tracking_number": order["tracking_number"],
        "tracking_url": order["tracking_url"],
        "shipping_address": order["shipping_address"]
    }

    log_audit(session_id, "get_order_status", {"order_id": req.order_id},
              result, (time.perf_counter() - t0) * 1000)
    return result


def _status_description(status: str) -> str:
    descriptions = {
        "processing": "Your order is being prepared and will be shipped soon.",
        "confirmed": "Your order has been confirmed and is being picked and packed.",
        "shipped": "Your order is on its way! Use the tracking link for updates.",
        "out_for_delivery": "Great news — your order is out for delivery today!",
        "delivered": "Your order has been delivered. Enjoy your NOVA goodies!",
        "returned": "Your return has been received and is being processed.",
        "refunded": "Your refund has been processed. Please allow 5-10 business days.",
        "cancelled": "This order has been cancelled. Contact us if this was unexpected."
    }
    return descriptions.get(status, "Status not available. Please contact support.")


# ── Tool 2: Process Return ────────────────────────────────────────────────────
@app.post("/tools/process_return")
async def process_return(req: ProcessReturnRequest):
    """
    Initiate a return and refund for an order.
    MCP Tool 2 of 5.
    """
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    order = DB["_order_index"].get(req.order_id)

    if not order:
        raise HTTPException(status_code=404, detail=f"Order {req.order_id} not found.")

    # Business rules
    if order["status"] not in ["delivered"]:
        result = {
            "success": False,
            "order_id": req.order_id,
            "reason": f"Returns are only accepted for delivered orders. Current status: {order['status']}."
        }
        log_audit(session_id, "process_return",
                  {"order_id": req.order_id, "reason": req.reason}, result,
                  (time.perf_counter() - t0) * 1000)
        return result

    if not order.get("return_eligible", True):
        result = {
            "success": False,
            "order_id": req.order_id,
            "reason": "This order is outside the 30-day return window."
        }
        log_audit(session_id, "process_return",
                  {"order_id": req.order_id, "reason": req.reason}, result,
                  (time.perf_counter() - t0) * 1000)
        return result

    # Calculate refund
    items_to_return = req.items or [i["product_id"] for i in order["items"]]
    refund_amount = sum(
        item["subtotal"] for item in order["items"]
        if item.get("product_id") in items_to_return or not req.items
    )

    return_label_id = f"RTN-{uuid.uuid4().hex[:8].upper()}"
    result = {
        "success": True,
        "order_id": req.order_id,
        "return_id": return_label_id,
        "return_label": f"https://returns.nova.com/{return_label_id}",
        "items_accepted": items_to_return,
        "refund_amount": round(refund_amount, 2),
        "refund_method": order["payment_method"],
        "estimated_refund_days": 5,
        "instructions": [
            "Print your return label from the link above.",
            "Pack items securely in original packaging if available.",
            "Drop off at any post office within 14 days.",
            f"Your refund of ${refund_amount:.2f} will arrive in 5 business days once received."
        ],
        "reason_logged": req.reason
    }

    log_audit(session_id, "process_return",
              {"order_id": req.order_id, "reason": req.reason, "items": items_to_return},
              result, (time.perf_counter() - t0) * 1000)
    return result


# ── Tool 3: Get Product Info ──────────────────────────────────────────────────
@app.post("/tools/get_product_info")
async def get_product_info(req: ProductInfoRequest):
    """
    Retrieve full product details including ingredients, sizing, and stock.
    MCP Tool 3 of 5.
    """
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    product = DB["_product_index"].get(req.product_id)

    if not product:
        # Try fuzzy match by name
        matches = [
            p for p in DB["products"]
            if req.product_id.lower() in p["name"].lower()
        ]
        if matches:
            product = matches[0]
        else:
            raise HTTPException(status_code=404, detail=f"Product {req.product_id} not found.")

    result = {
        "product_id": product["product_id"],
        "name": product["name"],
        "category": product["category"],
        "price": product["price"],
        "stock_status": "in_stock" if product.get("stock", 0) > 0 else "out_of_stock",
        "stock_count": product.get("stock", 0),
        "rating": product.get("rating"),
        "review_count": product.get("review_count"),
        "description": product.get("description", ""),
        "is_bestseller": product.get("is_bestseller", False),
        "ingredients": product.get("ingredients", []),
        "skin_types": product.get("skin_types", []),
        "concerns": product.get("concerns", []),
        "sizes": product.get("sizes", []),
        "material": product.get("material"),
        "volume_ml": product.get("volume_ml"),
        "spf": product.get("spf"),
        "is_vegan": random.random() > 0.3,  # Simulated field
        "is_cruelty_free": True,  # NOVA is always cruelty-free
        "is_sustainable": product.get("is_sustainable", random.random() > 0.6)
    }

    log_audit(session_id, "get_product_info", {"product_id": req.product_id},
              result, (time.perf_counter() - t0) * 1000)
    return result


# ── Tool 4: Get Customer History ──────────────────────────────────────────────
@app.post("/tools/get_customer_history")
async def get_customer_history(req: CustomerHistoryRequest):
    """
    Retrieve a customer's purchase history, preferences, and loyalty status.
    MCP Tool 4 of 5.
    """
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    customer = DB["_customer_index"].get(req.customer_id)

    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {req.customer_id} not found.")

    # Get their recent orders
    customer_orders = [
        o for o in DB["orders"]
        if o["customer_id"] == req.customer_id
    ][-5:]  # Last 5 orders

    # Get purchased product details
    purchased_products = []
    for pid in customer.get("purchase_history", [])[:10]:
        product = DB["_product_index"].get(pid)
        if product:
            purchased_products.append({
                "product_id": pid,
                "name": product["name"],
                "category": product["category"]
            })

    loyalty_benefits = {
        "bronze": "5% off all orders",
        "silver": "10% off + free standard shipping",
        "gold": "15% off + early access to new launches",
        "platinum": "20% off + dedicated personal stylist"
    }

    result = {
        "customer_id": customer["customer_id"],
        "name": customer["name"],
        "email": customer["email"],
        "country": customer["country"],
        "skin_type": customer["skin_type"],
        "hair_type": customer.get("hair_type"),
        "preferred_categories": customer.get("preferred_categories", []),
        "loyalty_tier": customer["loyalty_tier"],
        "loyalty_points": customer.get("loyalty_points", 0),
        "loyalty_benefits": loyalty_benefits.get(customer["loyalty_tier"], ""),
        "total_orders": customer.get("total_orders", len(customer_orders)),
        "total_spent": customer.get("total_spent", 0),
        "last_purchase_date": customer.get("last_purchase_date"),
        "is_vip": customer.get("is_vip", False),
        "recent_orders": [
            {"order_id": o["order_id"], "status": o["status"],
             "total": o["total_amount"], "date": o["order_date"][:10]}
            for o in customer_orders
        ],
        "purchased_products": purchased_products
    }

    log_audit(session_id, "get_customer_history", {"customer_id": req.customer_id},
              result, (time.perf_counter() - t0) * 1000)
    return result


# ── Tool 5: Recommend Products ────────────────────────────────────────────────
@app.post("/tools/recommend_products")
async def recommend_products(req: RecommendRequest):
    """
    Generate personalized product recommendations based on customer profile and context.
    MCP Tool 5 of 5.
    """
    t0 = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())[:8]

    customer = DB["_customer_index"].get(req.customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {req.customer_id} not found.")

    # Filter products not already purchased
    purchased_ids = set(customer.get("purchase_history", []))
    context_lower = req.context.lower()

    # Score products
    scored = []
    for product in DB["products"]:
        if product["product_id"] in purchased_ids:
            continue  # Skip already purchased

        score = 0.0

        # Category preference match
        if product["category"] in customer.get("preferred_categories", []):
            score += 2.0

        # Filter by requested category
        if req.category and product["category"] != req.category:
            continue

        # Skin type match (skincare/makeup)
        if product.get("skin_types") and customer.get("skin_type"):
            if customer["skin_type"] in product.get("skin_types", []):
                score += 1.5

        # Context matching
        if "oily" in context_lower and customer["skin_type"] == "oily":
            if "oily" in str(product.get("skin_types", [])).lower():
                score += 1.0
        if "dry" in context_lower and customer["skin_type"] == "dry":
            if "dry" in str(product.get("skin_types", [])).lower():
                score += 1.0
        if "gift" in context_lower and product.get("is_bestseller"):
            score += 1.0

        # Bestseller boost
        if product.get("is_bestseller"):
            score += 0.5

        # Rating boost
        score += product.get("rating", 3.5) * 0.2

        scored.append((score, product))

    # Sort and take top 5
    scored.sort(key=lambda x: x[0], reverse=True)
    top_5 = scored[:5]

    recommendations = []
    for score, product in top_5:
        reason = _build_recommendation_reason(product, customer, req.context)
        recommendations.append({
            "product_id": product["product_id"],
            "name": product["name"],
            "category": product["category"],
            "price": product["price"],
            "rating": product.get("rating"),
            "is_bestseller": product.get("is_bestseller", False),
            "recommendation_reason": reason,
            "relevance_score": round(score, 2)
        })

    result = {
        "customer_id": req.customer_id,
        "context": req.context,
        "recommendations": recommendations,
        "total_found": len(scored)
    }

    log_audit(session_id, "recommend_products",
              {"customer_id": req.customer_id, "context": req.context},
              result, (time.perf_counter() - t0) * 1000)
    return result


def _build_recommendation_reason(product: dict, customer: dict, context: str) -> str:
    """Build a human-readable recommendation reason."""
    reasons = []
    if product.get("is_bestseller"):
        reasons.append("one of our bestsellers")
    if customer.get("skin_type") in product.get("skin_types", []):
        reasons.append(f"great for {customer['skin_type']} skin")
    if product.get("rating", 0) >= 4.5:
        reasons.append(f"rated {product['rating']}/5 by customers")
    if product["category"] in customer.get("preferred_categories", []):
        reasons.append("matches your preferred categories")

    if not reasons:
        reasons.append("highly rated and new to you")

    return f"Recommended because it's {', '.join(reasons)}."


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "server": "NOVA MCP Server",
        "version": "1.0.0",
        "tools": ["get_order_status", "process_return", "get_product_info",
                  "get_customer_history", "recommend_products"],
        "db_loaded": bool(DB),
        "audit_log": str(AUDIT_LOG_PATH)
    }


@app.get("/tools")
async def list_tools():
    """List all available MCP tools with their schemas."""
    return {
        "tools": [
            {
                "name": "get_order_status",
                "description": "Get real-time order status, tracking, and delivery estimate",
                "endpoint": "/tools/get_order_status",
                "params": {"order_id": "string", "session_id": "string (optional)"}
            },
            {
                "name": "process_return",
                "description": "Initiate a return and generate refund",
                "endpoint": "/tools/process_return",
                "params": {"order_id": "string", "reason": "string",
                           "items": "list[string] (optional)", "session_id": "string (optional)"}
            },
            {
                "name": "get_product_info",
                "description": "Get full product details including ingredients and sizing",
                "endpoint": "/tools/get_product_info",
                "params": {"product_id": "string", "session_id": "string (optional)"}
            },
            {
                "name": "get_customer_history",
                "description": "Get customer purchase history, preferences, and loyalty status",
                "endpoint": "/tools/get_customer_history",
                "params": {"customer_id": "string", "session_id": "string (optional)"}
            },
            {
                "name": "recommend_products",
                "description": "Get personalised product recommendations",
                "endpoint": "/tools/recommend_products",
                "params": {"customer_id": "string", "context": "string",
                           "category": "string (optional)", "session_id": "string (optional)"}
            }
        ]
    }
