import time
from typing import Dict, Any, Optional

class OrderLookupError(Exception):
    """Base exception for order lookup failures."""
    pass

class OrderServiceTimeoutError(OrderLookupError):
    """Raised when the remote microservice or database times out."""
    pass

class OrderNotFoundError(OrderLookupError):
    """Raised when an order ID is not found in the database."""
    pass

# Mock Database
MOCK_ORDERS_DB: Dict[str, Dict[str, Any]] = {
    "ORD-88219": {
        "order_id": "ORD-88219",
        "customer_id": "CUST-9901",
        "plan": "Enterprise",
        "status": "active",
        "amount_usd": 1200.00,
        "renewal_date": "2026-11-01",
        "payment_method": "ACH / Wire Transfer",
        "seats_allocated": 150
    },
    "ORD-10492": {
        "order_id": "ORD-10492",
        "customer_id": "CUST-4122",
        "plan": "Professional",
        "status": "past_due",
        "amount_usd": 249.00,
        "renewal_date": "2026-08-15",
        "payment_method": "Credit Card (Visa ending 4012)",
        "seats_allocated": 25
    },
    "ORD-33104": {
        "order_id": "ORD-33104",
        "customer_id": "CUST-1033",
        "plan": "Starter",
        "status": "cancelled",
        "amount_usd": 49.00,
        "renewal_date": "2026-07-01",
        "payment_method": "Credit Card (Mastercard ending 9182)",
        "seats_allocated": 5
    }
}

def check_order_status(order_id: str, force_failure: bool = False, failure_type: str = "timeout") -> Dict[str, Any]:
    """
    Mock customer support tool to retrieve order, subscription, and billing status.
    
    Args:
        order_id: The order identifier (e.g., 'ORD-88219').
        force_failure: If True, deliberately triggers a failure (Failure Injection 3).
        failure_type: Either 'timeout' or 'crash'.
        
    Returns:
        Dictionary containing order and subscription details.
        
    Raises:
        OrderServiceTimeoutError: If timeout failure is simulated.
        OrderLookupError: If database crash or unhandled error is simulated.
        OrderNotFoundError: If order_id does not exist in DB.
    """
    cleaned_id = order_id.strip().upper()
    
    if force_failure:
        if failure_type == "timeout":
            raise OrderServiceTimeoutError(
                f"Timeout (504 Gateway Timeout): Order service downstream database connection timed out after 5000ms for {cleaned_id}."
            )
        else:
            raise OrderLookupError(
                f"Internal 500 DB Crash: Deadlock detected in cluster shard replica-02 when querying order {cleaned_id}."
            )
            
    if cleaned_id in MOCK_ORDERS_DB:
        return MOCK_ORDERS_DB[cleaned_id]
        
    raise OrderNotFoundError(f"Order '{cleaned_id}' not found in CloudFlow order registry.")
