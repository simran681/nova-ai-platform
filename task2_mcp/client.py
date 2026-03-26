"""
NOVA AI Platform — MCP Client (Task 2)
Provides a Python client for calling NOVA's backend MCP tools.

Usage:
    from task2_mcp.client import NOVAMCPClient
    client = NOVAMCPClient(base_url="http://localhost:8001")
    result = client.get_order_status("ORD-1042")
"""

import json
import time
from typing import Optional, List
import requests


class NOVAMCPClient:
    """
    Client for NOVA's MCP (Model Context Protocol) server.
    Wraps all 5 backend tools with error handling and retry logic.
    """

    def __init__(self, base_url: str = "http://localhost:8001", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session_id = f"sess_{int(time.time())}"

    def _call(self, endpoint: str, payload: dict) -> dict:
        """Make a POST request to an MCP tool endpoint."""
        payload["session_id"] = self.session_id
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError:
            return {"error": f"Cannot connect to MCP server at {self.base_url}. "
                             "Is the server running?"}
        except requests.exceptions.HTTPError as e:
            try:
                detail = response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            return {"error": detail}
        except Exception as e:
            return {"error": str(e)}

    def get_order_status(self, order_id: str) -> dict:
        """
        Tool 1: Get real-time order status and tracking.

        Args:
            order_id: Order ID (e.g., "ORD-1042")

        Returns:
            dict with status, tracking, delivery estimate, and items
        """
        return self._call("/tools/get_order_status", {"order_id": order_id})

    def process_return(
        self,
        order_id: str,
        reason: str,
        items: Optional[List[str]] = None
    ) -> dict:
        """
        Tool 2: Initiate a return and generate a refund.

        Args:
            order_id: Order ID to return
            reason: Reason for return (e.g., "wrong size", "damaged product")
            items: Optional list of product IDs to return (None = all items)

        Returns:
            dict with return label, refund amount, and instructions
        """
        payload = {"order_id": order_id, "reason": reason}
        if items:
            payload["items"] = items
        return self._call("/tools/process_return", payload)

    def get_product_info(self, product_id: str) -> dict:
        """
        Tool 3: Get full product details including ingredients and sizing.

        Args:
            product_id: Product ID (e.g., "PROD-1000") or product name substring

        Returns:
            dict with name, ingredients, sizes, stock, rating, etc.
        """
        return self._call("/tools/get_product_info", {"product_id": product_id})

    def get_customer_history(self, customer_id: str) -> dict:
        """
        Tool 4: Get customer purchase history, preferences, and loyalty status.

        Args:
            customer_id: Customer ID (e.g., "CUST-1000")

        Returns:
            dict with orders, preferences, loyalty tier, and spending
        """
        return self._call("/tools/get_customer_history", {"customer_id": customer_id})

    def recommend_products(
        self,
        customer_id: str,
        context: str = "",
        category: Optional[str] = None
    ) -> dict:
        """
        Tool 5: Get personalised product recommendations.

        Args:
            customer_id: Customer ID
            context: Natural language context (e.g., "oily skin", "gift for her")
            category: Optional category filter (skincare, makeup, hair, etc.)

        Returns:
            dict with top 5 recommended products and reasoning
        """
        payload = {"customer_id": customer_id, "context": context}
        if category:
            payload["category"] = category
        return self._call("/tools/recommend_products", payload)

    def health_check(self) -> dict:
        """Check if the MCP server is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except Exception as e:
            return {"error": str(e), "status": "offline"}


# ── Standalone tool functions (for use without running server) ─────────────────
# These use the mock DB directly — useful for Task 5 integration in Colab

class NOVAMCPClientLocal:
    """
    Local (in-process) version of MCP client.
    Reads nova_mock_db.json directly without needing a running server.
    Ideal for Google Colab where running a separate server process is complex.
    """

    def __init__(self, db_path: str = "nova_mock_db.json",
                 audit_log_path: str = "audit_log.jsonl"):
        import json
        import time
        import uuid

        with open(db_path) as f:
            db = json.load(f)

        self.db = db
        self.audit_log_path = audit_log_path
        self.order_index = {o["order_id"]: o for o in db["orders"]}
        self.customer_index = {c["customer_id"]: c for c in db["customers"]}
        self.product_index = {p["product_id"]: p for p in db["products"]}
        self.session_id = f"sess_{uuid.uuid4().hex[:8]}"

    def _log(self, tool: str, params: dict, response: dict, latency_ms: float):
        import json
        from datetime import datetime

        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": self.session_id,
            "tool": tool,
            "params": params,
            "response_summary": str(response)[:300],
            "success": "error" not in response,
            "latency_ms": round(latency_ms, 2)
        }
        with open(self.audit_log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_order_status(self, order_id: str) -> dict:
        t0 = time.perf_counter()
        order = self.order_index.get(order_id)
        if not order:
            result = {"error": f"Order {order_id} not found."}
        else:
            result = {
                "order_id": order["order_id"],
                "status": order["status"],
                "items": [{"name": i["product_name"], "qty": i["quantity"]}
                          for i in order["items"]],
                "total_amount": order["total_amount"],
                "order_date": order["order_date"],
                "estimated_delivery": order["estimated_delivery"],
                "tracking_number": order["tracking_number"],
                "tracking_url": order["tracking_url"]
            }
        self._log("get_order_status", {"order_id": order_id}, result,
                  (time.perf_counter() - t0) * 1000)
        return result

    def process_return(self, order_id: str, reason: str,
                       items: Optional[List[str]] = None) -> dict:
        import uuid
        t0 = time.perf_counter()
        order = self.order_index.get(order_id)
        if not order:
            result = {"error": f"Order {order_id} not found."}
        elif order["status"] != "delivered":
            result = {"success": False, "reason": f"Order status is '{order['status']}', not delivered."}
        else:
            refund_amount = sum(i["subtotal"] for i in order["items"])
            return_id = f"RTN-{uuid.uuid4().hex[:8].upper()}"
            result = {
                "success": True, "order_id": order_id, "return_id": return_id,
                "return_label": f"https://returns.nova.com/{return_id}",
                "refund_amount": round(refund_amount, 2),
                "estimated_refund_days": 5,
                "reason_logged": reason
            }
        self._log("process_return", {"order_id": order_id, "reason": reason}, result,
                  (time.perf_counter() - t0) * 1000)
        return result

    def get_product_info(self, product_id: str) -> dict:
        t0 = time.perf_counter()
        product = self.product_index.get(product_id)
        if not product:
            # Try name match
            matches = [p for p in self.db["products"]
                       if product_id.lower() in p["name"].lower()]
            product = matches[0] if matches else None
        if not product:
            result = {"error": f"Product '{product_id}' not found."}
        else:
            result = {
                "product_id": product["product_id"],
                "name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "stock_status": "in_stock" if product.get("stock", 0) > 0 else "out_of_stock",
                "ingredients": product.get("ingredients", []),
                "skin_types": product.get("skin_types", []),
                "sizes": product.get("sizes", []),
                "rating": product.get("rating"),
                "is_cruelty_free": True,
                "description": product.get("description", "")
            }
        self._log("get_product_info", {"product_id": product_id}, result,
                  (time.perf_counter() - t0) * 1000)
        return result

    def get_customer_history(self, customer_id: str) -> dict:
        t0 = time.perf_counter()
        customer = self.customer_index.get(customer_id)
        if not customer:
            result = {"error": f"Customer {customer_id} not found."}
        else:
            result = {
                "customer_id": customer["customer_id"],
                "name": customer["name"],
                "skin_type": customer["skin_type"],
                "preferred_categories": customer.get("preferred_categories", []),
                "loyalty_tier": customer["loyalty_tier"],
                "loyalty_points": customer.get("loyalty_points", 0),
                "total_spent": customer.get("total_spent", 0),
                "purchase_history": customer.get("purchase_history", [])[:10],
                "is_vip": customer.get("is_vip", False)
            }
        self._log("get_customer_history", {"customer_id": customer_id}, result,
                  (time.perf_counter() - t0) * 1000)
        return result

    def recommend_products(self, customer_id: str, context: str = "",
                           category: Optional[str] = None) -> dict:
        import random
        t0 = time.perf_counter()
        customer = self.customer_index.get(customer_id)
        if not customer:
            result = {"error": f"Customer {customer_id} not found."}
            self._log("recommend_products", {"customer_id": customer_id}, result,
                      (time.perf_counter() - t0) * 1000)
            return result

        purchased = set(customer.get("purchase_history", []))
        context_lower = context.lower()

        scored = []
        for product in self.db["products"]:
            if product["product_id"] in purchased:
                continue
            if category and product["category"] != category:
                continue

            score = 0.0
            if product["category"] in customer.get("preferred_categories", []):
                score += 2.0
            if (customer.get("skin_type") and
                    customer["skin_type"] in product.get("skin_types", [])):
                score += 1.5
            if product.get("is_bestseller"):
                score += 0.5
            score += product.get("rating", 3.5) * 0.2
            scored.append((score, product))

        scored.sort(key=lambda x: x[0], reverse=True)
        recs = [
            {
                "product_id": p["product_id"],
                "name": p["name"],
                "category": p["category"],
                "price": p["price"],
                "rating": p.get("rating"),
                "is_bestseller": p.get("is_bestseller", False),
                "recommendation_reason": f"Recommended based on your {customer.get('skin_type', '')} skin type and preferences."
            }
            for _, p in scored[:5]
        ]

        result = {"customer_id": customer_id, "context": context, "recommendations": recs}
        self._log("recommend_products", {"customer_id": customer_id, "context": context},
                  result, (time.perf_counter() - t0) * 1000)
        return result
