"""Column name compatibility layer."""

# Mapping of old column names to new ones
COLUMN_MAPPINGS = {
    "options.contracts": {
        "strike": "strike_price",
        "bid": "bid_price", 
        "ask": "ask_price",
    },
    "market.price_data": {
        # Add any other mappings here
    }
}


def map_column_names(table: str, columns: list) -> list:
    """Map old column names to new ones."""
    if table not in COLUMN_MAPPINGS:
        return columns
        
    mapping = COLUMN_MAPPINGS[table]
    return [mapping.get(col, col) for col in columns]


def get_compatible_query(query: str) -> str:
    """Update query with compatible column names."""
    # Simple replacement - in production use SQL parser
    for table, mappings in COLUMN_MAPPINGS.items():
        if table in query:
            for old, new in mappings.items():
                # Replace column names in SELECT, WHERE, ORDER BY
                query = query.replace(f" {old} ", f" {new} ")
                query = query.replace(f" {old},", f" {new},")
                query = query.replace(f".{old} ", f".{new} ")
                query = query.replace(f".{old},", f".{new},")
                
    return query
