"""In-memory knowledge base registry."""
from __future__ import annotations
from pathlib import Path

KB_DIR = Path(__file__).parent.parent / "knowledge_base"

KB_SUMMARIES: dict[str, str] = {
    "kb-001": "Shipping zones with standard and express timelines, ship-after-order timing, express fee/free threshold, and lost-parcel trigger after 5+ business days past estimate.",
    "kb-002": "30-day return window from delivery, eligible and excluded items, return start steps with 14-day drop-off, EU free returns, and refund timing after receipt.",
    "kb-003": "Refund issuance after inspection, payment-method landing windows, bank hold guidance, partial refund reasons, and damaged-in-transit refund rule.",
    "kb-004": "Damaged or faulty arrivals: photo requirement within 48 hours, no-return resolution, replacement vs refund, 14-day claim limit, and handoff to warranty after that.",
    "kb-005": "Two-year manufacturer warranty from delivery, coverage and exclusions, optional extended warranty, claim steps, and repair or replacement outcomes.",
    "kb-006": "Cancellation rules pre- and post-shipment, refuse-delivery option, return-after-delivery path, partial cancellations, and refund timing alignment with refund policy.",
    "kb-007": "Order change windows, self-edit within 15 minutes, support-assisted changes before shipping, limits on item swaps, and carrier redirect after shipment.",
    "kb-008": "Accepted payment methods, charge timing by method, declined payment hold and cancel window, and saved card handling with PCI processor.",
    "kb-009": "Account access help: password reset steps, email-not-found guidance, email change verification, 2FA support, and account deletion timeline.",
    "kb-010": "Tracking availability timing, normal scan delays, lost-parcel threshold and investigation timing, delivered-but-missing steps, and international tracking gaps.",
    "kb-011": "International shipping coverage, duties and VAT responsibility, DDU vs DDP behavior, refused or unclaimed parcel refunds, and restricted item limits.",
    "kb-012": "Gift card purchase options, redemption and balance checks, five-year expiry, loss and compromise handling, and refund priority when mixed payments are used.",
    "kb-013": "Sizing resources, how to measure, between-size guidance, returns for size mismatch, and footwear conversion notes with no half sizes.",
    "kb-014": "Subscription cadence controls, skip or pause options, cancellation rules, renewal notifications, and failed payment retry behavior.",
    "kb-015": "Promo code application, one-code limit, rejection reasons, retroactive window of 48 hours, and stacking restrictions with sale or subscription discounts.",
    "kb-016": "Product care basics by category: clothing, knitwear, leather, electronics, and home goods, with care-label precedence and contact support for unclear labels.",
    "kb-017": "Store credit account balances, opt-out checkout usage settings, 24-month validity limits, and authorization rules allowing store credit to be issued as goodwill compensation for service issues (e.g., significantly delayed shipments).",
    "kb-018": "Support channels and response times, required ticket details, escalation keyword after 3 business days, and supported languages.",
}

KB_FILES: dict[str, str] = {
    "kb-001": "kb-001-shipping-times.md",
    "kb-002": "kb-002-return-policy.md",
    "kb-003": "kb-003-refunds-processing.md",
    "kb-004": "kb-004-damaged-faulty.md",
    "kb-005": "kb-005-warranty.md",
    "kb-006": "kb-006-cancellation.md",
    "kb-007": "kb-007-order-changes.md",
    "kb-008": "kb-008-payment-methods.md",
    "kb-009": "kb-009-account-login.md",
    "kb-010": "kb-010-tracking.md",
    "kb-011": "kb-011-international-customs.md",
    "kb-012": "kb-012-gift-cards.md",
    "kb-013": "kb-013-size-fit.md",
    "kb-014": "kb-014-subscriptions.md",
    "kb-015": "kb-015-promo-codes.md",
    "kb-016": "kb-016-product-care.md",
    "kb-017": "kb-017-store-credit.md",
    "kb-018": "kb-018-contact-support.md",
}


def load_articles(article_ids: list[str]) -> str:
    """Load the requested articles and preserve missing-file markers."""
    chunks: list[str] = []
    for aid in article_ids:
        filename = KB_FILES.get(aid)
        if not filename:
            chunks.append(f"[{aid}] — Article ID not recognised.")
            continue
        filepath = KB_DIR / filename
        if filepath.exists():
            chunks.append(f"--- {aid} ---\n{filepath.read_text(encoding='utf-8')}")
        else:
            chunks.append(
                f"--- {aid} ---\n[Article file not found on disk. "
                f"Summary: {KB_SUMMARIES.get(aid, 'No summary available.')}]"
            )
    return "\n\n".join(chunks)


KEYWORD_TO_KB: dict[str, set[str]] = {
    "refund": {"kb-003", "kb-002", "kb-004"},
    "refunds": {"kb-003", "kb-002", "kb-004"},
    "return": {"kb-002"},
    "returns": {"kb-002"},
    "damaged": {"kb-004"},
    "faulty": {"kb-004"},
    "warranty": {"kb-005"},
    "cancel": {"kb-006"},
    "cancellation": {"kb-006"},
    "change": {"kb-007"},
    "edit": {"kb-007"},
    "payment": {"kb-008"},
    "card": {"kb-008"},
    "login": {"kb-009"},
    "password": {"kb-009"},
    "2fa": {"kb-009"},
    "account": {"kb-009"},
    "tracking": {"kb-010"},
    "track": {"kb-010"},
    "delivered": {"kb-010", "kb-001"},
    "shipping": {"kb-001", "kb-010", "kb-011"},
    "customs": {"kb-011"},
    "international": {"kb-011"},
    "gift": {"kb-012"},
    "size": {"kb-013"},
    "fit": {"kb-013"},
    "subscription": {"kb-014"},
    "promo": {"kb-015"},
    "code": {"kb-015"},
    "care": {"kb-016"},
    "store credit": {"kb-017"},
    "credit": {"kb-017"},
    "support": {"kb-018"},
    "contact": {"kb-018"},
}


def summaries_for_planner(ticket_text: str) -> str:
    """Return the filtered summary table exposed to the planner prompt."""
    selected_ids: set[str] = set()
    text = ticket_text.lower()
    for keyword, ids in KEYWORD_TO_KB.items():
        if keyword in text:
            selected_ids.update(ids)

    if not selected_ids:
        selected_ids = set(KB_SUMMARIES.keys())

    lines = ["Available knowledge base articles (id: one-line summary):"]
    for aid in sorted(selected_ids):
        summary = KB_SUMMARIES.get(aid)
        if summary:
            lines.append(f"  {aid}: {summary}")
    return "\n".join(lines)
