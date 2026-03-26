"""
NOVA AI Platform - Synthetic Data Generator
Generates nova_mock_db.json using Python Faker.
Run: python generate_mock_db.py
"""

import json
import random
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
NUM_CUSTOMERS = 50
NUM_ORDERS = 100
NUM_PRODUCTS = 200

CATEGORIES = ["skincare", "makeup", "hair", "apparel", "footwear", "accessories"]
SKIN_TYPES = ["oily", "dry", "combination", "sensitive", "normal"]
LOYALTY_TIERS = ["bronze", "silver", "gold", "platinum"]

ORDER_STATUSES = [
    "processing", "confirmed", "shipped", "out_for_delivery",
    "delivered", "returned", "refunded", "cancelled"
]

# ── Product catalog templates ────────────────────────────────────────────────
PRODUCT_TEMPLATES = {
    "skincare": {
        "names": [
            "Glow Serum", "Hydra Boost Moisturizer", "Vitamin C Brightening Cream",
            "Retinol Night Repair", "SPF 50 Daily Sunscreen", "Gentle Foam Cleanser",
            "Hyaluronic Acid Essence", "Niacinamide Pore Minimizer", "AHA/BHA Exfoliant",
            "Peptide Eye Cream", "Ceramide Barrier Repair", "Squalane Face Oil",
            "Clay Detox Mask", "Sheet Mask Bundle", "Micellar Cleansing Water",
            "Toner Mist", "Bakuchiol Serum", "Collagen Plumping Serum",
            "Green Tea Antioxidant Cream", "Snail Mucin Essence"
        ],
        "ingredients_pool": [
            "Hyaluronic Acid", "Niacinamide", "Vitamin C (Ascorbic Acid)", "Retinol",
            "Ceramides", "Peptides", "Squalane", "Glycerin", "Aloe Vera",
            "Green Tea Extract", "Salicylic Acid", "Lactic Acid", "Glycolic Acid",
            "Zinc Oxide", "Titanium Dioxide", "Ferulic Acid", "Bakuchiol",
            "Snail Secretion Filtrate", "Adenosine", "Centella Asiatica"
        ],
        "concerns": ["acne", "aging", "hyperpigmentation", "dryness", "oiliness", "sensitivity"]
    },
    "makeup": {
        "names": [
            "Velvet Matte Foundation", "Dewy Skin Tinted Moisturizer", "Concealer Pro",
            "Setting Powder", "Bronzer Glow Palette", "Blush Duo", "Highlighter",
            "Eyeshadow Palette 12-Pan", "Waterproof Mascara", "Brow Gel",
            "Lip Gloss Set", "Matte Liquid Lipstick", "Lip Liner Pencil",
            "Eyeliner Pen", "Setting Spray", "Primer Blur", "CC Cream SPF30",
            "Eyeshadow Primer", "Contour Stick", "Nude Lip Kit"
        ],
        "ingredients_pool": [
            "Dimethicone", "Talc", "Mica", "Iron Oxides", "Titanium Dioxide",
            "Vitamin E (Tocopherol)", "Jojoba Oil", "Shea Butter", "Beeswax",
            "Carnauba Wax", "Kaolin", "Silica", "Hyaluronic Acid"
        ],
        "concerns": ["coverage", "longevity", "color_payoff", "skin_tone_match"]
    },
    "hair": {
        "names": [
            "Repair & Restore Shampoo", "Moisture Surge Conditioner", "Hair Mask",
            "Leave-in Conditioner Spray", "Heat Protectant Serum", "Scalp Scrub",
            "Anti-Frizz Smoothing Cream", "Volume Mousse", "Hair Oil Blend",
            "Bond Repair Treatment", "Dry Shampoo", "Color Protect Shampoo",
            "Curl Defining Gel", "Keratin Smoothing Mask", "Purple Toning Shampoo"
        ],
        "ingredients_pool": [
            "Keratin", "Argan Oil", "Biotin", "Panthenol", "Hydrolyzed Silk",
            "Coconut Oil", "Castor Oil", "Rosemary Extract", "Caffeine",
            "Niacinamide", "Glycerin", "Cetyl Alcohol"
        ],
        "concerns": ["damage", "frizz", "volume", "color_protection", "scalp_health"]
    },
    "apparel": {
        "names": [
            "Everyday Comfort Tee", "Relaxed Linen Blazer", "High-Rise Straight Jeans",
            "Wrap Midi Dress", "Cropped Utility Jacket", "Wide-Leg Trousers",
            "Ribbed Knit Sweater", "Oversized Hoodie", "Slip Dress",
            "Button-Down Shirt", "Co-ord Set", "Bodysuit", "Mini Skirt",
            "Cargo Pants", "Turtleneck Top"
        ],
        "sizes": ["XS", "S", "M", "L", "XL", "XXL"],
        "materials": ["Cotton", "Linen", "Polyester", "Viscose", "Denim", "Wool blend"]
    },
    "footwear": {
        "names": [
            "Classic White Sneakers", "Barely-There Heeled Sandals", "Ankle Boots",
            "Platform Loafers", "Running Trainers", "Leather Ballet Flats",
            "Strappy Heels", "Chelsea Boots", "Mule Slides", "Chunky Sneakers",
            "Knee-High Boots", "Espadrilles", "Dad Sandals", "Block Heel Mules"
        ],
        "sizes": ["35", "36", "37", "38", "39", "40", "41", "42"],
        "materials": ["Leather", "Faux Leather", "Canvas", "Suede", "Mesh"]
    },
    "accessories": {
        "names": [
            "Gold Chain Necklace", "Pearl Drop Earrings", "Leather Mini Bag",
            "Canvas Tote", "Silk Scrunchie Set", "Sunglasses", "Watch",
            "Hair Clip Set", "Beaded Bracelet", "Belt", "Crossbody Bag",
            "Bucket Hat", "Hoop Earrings", "Charm Bracelet", "Coin Purse"
        ],
        "materials": ["Gold-plated", "Sterling Silver", "Leather", "Canvas", "Acrylic"]
    }
}


def generate_products():
    products = []
    product_id = 1000

    for category in CATEGORIES:
        template = PRODUCT_TEMPLATES[category]
        names = template["names"]

        for i, name in enumerate(names):
            product = {
                "product_id": f"PROD-{product_id}",
                "name": f"NOVA {name}",
                "category": category,
                "price": round(random.uniform(12.99, 89.99), 2),
                "stock": random.randint(0, 500),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "review_count": random.randint(10, 2000),
                "is_bestseller": random.random() < 0.2,
                "description": f"NOVA's {name} — crafted for your unique {category} needs.",
                "sku": f"NOVA-{category[:3].upper()}-{product_id}"
            }

            # Category-specific fields
            if category == "skincare":
                ingredients = random.sample(template["ingredients_pool"], k=random.randint(4, 8))
                product["ingredients"] = ingredients
                product["skin_types"] = random.sample(SKIN_TYPES, k=random.randint(2, 5))
                product["concerns"] = random.sample(template["concerns"], k=random.randint(1, 3))
                product["volume_ml"] = random.choice([30, 50, 100, 150, 200])
                product["spf"] = random.choice([None, None, None, 15, 30, 50])

            elif category == "makeup":
                ingredients = random.sample(template["ingredients_pool"], k=random.randint(3, 6))
                product["ingredients"] = ingredients
                product["skin_types"] = random.sample(SKIN_TYPES, k=random.randint(2, 5))
                product["shades"] = random.randint(1, 40)
                product["finish"] = random.choice(["matte", "dewy", "satin", "glitter"])

            elif category == "hair":
                ingredients = random.sample(template["ingredients_pool"], k=random.randint(3, 7))
                product["ingredients"] = ingredients
                product["hair_types"] = random.sample(
                    ["straight", "wavy", "curly", "coily", "fine", "thick"], k=3
                )
                product["concerns"] = random.sample(template["concerns"], k=2)
                product["volume_ml"] = random.choice([100, 200, 300, 400])

            elif category == "apparel":
                product["sizes"] = template["sizes"]
                product["material"] = random.choice(template["materials"])
                product["care_instructions"] = "Machine wash cold, tumble dry low"
                product["is_sustainable"] = random.random() < 0.3

            elif category == "footwear":
                product["sizes"] = template["sizes"]
                product["material"] = random.choice(template["materials"])
                product["heel_height_cm"] = random.choice([0, 0, 2, 4, 6, 8, 10])

            elif category == "accessories":
                product["material"] = random.choice(template["materials"])
                product["dimensions"] = f"{random.randint(10,30)}cm x {random.randint(5,20)}cm"

            products.append(product)
            product_id += 1

    # Pad remaining to reach NUM_PRODUCTS
    while len(products) < NUM_PRODUCTS:
        cat = random.choice(CATEGORIES)
        template = PRODUCT_TEMPLATES[cat]
        base_name = random.choice(template["names"])
        product_id += 1
        products.append({
            "product_id": f"PROD-{product_id}",
            "name": f"NOVA {base_name} Pro",
            "category": cat,
            "price": round(random.uniform(12.99, 89.99), 2),
            "stock": random.randint(0, 300),
            "rating": round(random.uniform(3.5, 5.0), 1),
            "review_count": random.randint(5, 500),
            "is_bestseller": False,
            "description": f"NOVA's premium {base_name} — elevated formula.",
            "sku": f"NOVA-{cat[:3].upper()}-{product_id}",
            "ingredients": random.sample(
                PRODUCT_TEMPLATES.get(cat, PRODUCT_TEMPLATES["skincare"])
                .get("ingredients_pool", ["Glycerin", "Water"]), k=3
            ) if "ingredients_pool" in PRODUCT_TEMPLATES.get(cat, {}) else [],
            "skin_types": random.sample(SKIN_TYPES, k=3) if cat in ["skincare", "makeup"] else []
        })

    return products[:NUM_PRODUCTS]


def generate_customers(products):
    customers = []
    for i in range(NUM_CUSTOMERS):
        product_ids = [p["product_id"] for p in products]
        purchase_history = random.sample(product_ids, k=random.randint(2, 15))

        customer = {
            "customer_id": f"CUST-{1000 + i}",
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "country": random.choice([
                "US", "UK", "CA", "AU", "DE", "FR", "IN", "SG",
                "AE", "NZ", "IE", "NL", "SE", "NO", "DK"
            ]),
            "skin_type": random.choice(SKIN_TYPES),
            "hair_type": random.choice(["straight", "wavy", "curly", "coily"]),
            "preferred_categories": random.sample(CATEGORIES, k=random.randint(1, 4)),
            "purchase_history": purchase_history,
            "total_orders": random.randint(1, 30),
            "total_spent": round(random.uniform(50, 2000), 2),
            "loyalty_tier": random.choice(LOYALTY_TIERS),
            "loyalty_points": random.randint(0, 5000),
            "registered_date": fake.date_between(start_date="-3y", end_date="today").isoformat(),
            "last_purchase_date": fake.date_between(start_date="-6m", end_date="today").isoformat(),
            "notes": "",
            "is_vip": random.random() < 0.15
        }
        customers.append(customer)
    return customers


def generate_orders(customers, products):
    orders = []
    product_map = {p["product_id"]: p for p in products}

    for i in range(NUM_ORDERS):
        customer = random.choice(customers)
        num_items = random.randint(1, 4)
        items = []

        for _ in range(num_items):
            product = random.choice(products)
            qty = random.randint(1, 3)
            items.append({
                "product_id": product["product_id"],
                "product_name": product["name"],
                "quantity": qty,
                "unit_price": product["price"],
                "subtotal": round(product["price"] * qty, 2)
            })

        order_date = fake.date_time_between(start_date="-1y", end_date="now")
        status = random.choice(ORDER_STATUSES)

        # Determine delivery estimate based on status
        if status in ["delivered", "returned", "refunded"]:
            delivery_date = order_date + timedelta(days=random.randint(3, 10))
            estimated_delivery = delivery_date.strftime("%Y-%m-%d")
        elif status in ["shipped", "out_for_delivery"]:
            delivery_date = datetime.now() + timedelta(days=random.randint(1, 5))
            estimated_delivery = delivery_date.strftime("%Y-%m-%d")
        else:
            delivery_date = datetime.now() + timedelta(days=random.randint(5, 14))
            estimated_delivery = delivery_date.strftime("%Y-%m-%d")

        total = round(sum(item["subtotal"] for item in items), 2)
        shipping = 0 if total > 50 else 5.99
        discount = round(total * random.choice([0, 0, 0, 0.1, 0.15, 0.2]), 2)

        order = {
            "order_id": f"ORD-{1000 + i}",
            "customer_id": customer["customer_id"],
            "customer_name": customer["name"],
            "status": status,
            "items": items,
            "subtotal": total,
            "shipping_fee": shipping,
            "discount_applied": discount,
            "total_amount": round(total + shipping - discount, 2),
            "payment_method": random.choice(["credit_card", "paypal", "apple_pay", "klarna"]),
            "shipping_address": {
                "street": fake.street_address(),
                "city": fake.city(),
                "country": customer["country"],
                "zip": fake.postcode()
            },
            "order_date": order_date.strftime("%Y-%m-%d %H:%M:%S"),
            "estimated_delivery": estimated_delivery,
            "tracking_number": f"TRK{fake.numerify('##########')}" if status != "processing" else None,
            "tracking_url": f"https://track.nova.com/orders/ORD-{1000 + i}" if status != "processing" else None,
            "return_eligible": status == "delivered" and (
                datetime.strptime(estimated_delivery, "%Y-%m-%d") > datetime.now() - timedelta(days=30)
            ),
            "return_window_days": 30,
            "notes": ""
        }
        orders.append(order)

    return orders


def generate_faqs():
    return [
        {
            "id": "FAQ-001",
            "category": "returns",
            "question": "What is NOVA's return policy?",
            "answer": "NOVA offers a 30-day hassle-free return policy. Items must be unused, unworn, and in original packaging. Skincare and makeup items are final sale once opened for hygiene reasons, unless defective."
        },
        {
            "id": "FAQ-002",
            "category": "shipping",
            "question": "How long does standard shipping take?",
            "answer": "Standard shipping takes 5-7 business days. Express shipping (2-3 days) is available for $9.99. Orders over $50 qualify for free standard shipping. International orders may take 10-14 business days."
        },
        {
            "id": "FAQ-003",
            "category": "shipping",
            "question": "Do you ship internationally?",
            "answer": "Yes! NOVA ships to 14 countries including US, UK, CA, AU, DE, FR, IN, SG, AE, NZ, IE, NL, SE, NO, and DK. International shipping rates are calculated at checkout."
        },
        {
            "id": "FAQ-004",
            "category": "loyalty",
            "question": "How does the NOVA loyalty program work?",
            "answer": "Earn 1 point per $1 spent. Bronze (0-499 pts): 5% off. Silver (500-1499 pts): 10% off + free shipping. Gold (1500-4999 pts): 15% off + early access. Platinum (5000+ pts): 20% off + personal stylist."
        },
        {
            "id": "FAQ-005",
            "category": "products",
            "question": "Are NOVA products cruelty-free?",
            "answer": "Yes, all NOVA products are 100% cruelty-free. We never test on animals and are certified by Leaping Bunny. Over 60% of our range is also vegan-friendly, clearly labelled on each product page."
        },
        {
            "id": "FAQ-006",
            "category": "sizing",
            "question": "How do I find my correct clothing size?",
            "answer": "Use our size guide: XS (UK 6-8), S (UK 8-10), M (UK 10-12), L (UK 12-14), XL (UK 14-16), XXL (UK 16-18). Measure your bust, waist, and hips and compare to the size chart. If between sizes, size up for comfort."
        },
        {
            "id": "FAQ-007",
            "category": "sizing",
            "question": "How do I find my shoe size?",
            "answer": "NOVA footwear follows EU sizing: 35 (UK 2.5), 36 (UK 3.5), 37 (UK 4), 38 (UK 5), 39 (UK 6), 40 (UK 6.5), 41 (UK 7.5), 42 (UK 8). Measure your foot length in cm and match to the chart."
        },
        {
            "id": "FAQ-008",
            "category": "payments",
            "question": "What payment methods does NOVA accept?",
            "answer": "NOVA accepts Visa, Mastercard, Amex, PayPal, Apple Pay, Google Pay, and Klarna (Buy Now Pay Later in 3 interest-free instalments)."
        }
    ]


def main():
    print("Generating NOVA synthetic database...")

    products = generate_products()
    print(f"  Generated {len(products)} products")

    customers = generate_customers(products)
    print(f"  Generated {len(customers)} customers")

    orders = generate_orders(customers, products)
    print(f"  Generated {len(orders)} orders")

    faqs = generate_faqs()
    print(f"  Generated {len(faqs)} FAQs")

    db = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "description": "NOVA AI Platform - Synthetic mock database",
            "counts": {
                "products": len(products),
                "customers": len(customers),
                "orders": len(orders),
                "faqs": len(faqs)
            }
        },
        "products": products,
        "customers": customers,
        "orders": orders,
        "faqs": faqs
    }

    output_path = "nova_mock_db.json"
    with open(output_path, "w") as f:
        json.dump(db, f, indent=2)

    print(f"\nSaved to {output_path}")
    print(f"File size: {len(json.dumps(db)) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
