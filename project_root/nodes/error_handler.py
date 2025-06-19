from typing import Dict, Any


def handle(state: Dict[str, Any]) -> Dict[str, Any]:
    """Simple error handler that logs the error message."""
    error = state.get("error")
    if error:
        print(f"[ErrorHandler] {error}")
    return state
