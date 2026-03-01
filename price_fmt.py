"""
Price formatting utility.
Automatically selects the right number of decimal places
so micro-price coins like PEPE (0.000007) display correctly.
"""

def fmt(price: float) -> str:
    """Format a price with enough significant figures regardless of magnitude."""
    if price == 0:
        return "0.0"
    if price >= 1000:
        return f"{price:,.2f}"
    if price >= 1:
        return f"{price:.4f}"
    if price >= 0.01:
        return f"{price:.6f}"
    if price >= 0.0001:
        return f"{price:.8f}"
    # Very small — use scientific or enough decimals
    # Find how many leading zeros after decimal point
    import math
    mag = -int(math.floor(math.log10(abs(price))))
    decimals = mag + 4  # show 4 significant digits after leading zeros
    return f"{price:.{decimals}f}"

def fmt_pct(value: float) -> str:
    return f"{value*100:.3f}%"

def fmt_size(size: float) -> str:
    """Format a base currency size (could be huge like PEPE or tiny like BTC)."""
    if size >= 1000:
        return f"{size:,.2f}"
    if size >= 1:
        return f"{size:.4f}"
    return f"{size:.8f}"
