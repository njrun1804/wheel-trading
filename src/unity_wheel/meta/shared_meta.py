"""
Shared MetaPrime singleton to prevent multiple instances across modules.
This ensures only one MetaPrime instance exists throughout the application.
"""

from datetime import datetime
from typing import Optional

# Global singleton instance
_meta_instance: Optional[object] = None


def get_shared_meta():
    """Get or create the shared MetaPrime instance."""
    import os
    
    # Check if meta system is disabled
    if os.environ.get('DISABLE_META_AUTOSTART') == '1':
        # Return mock instance when disabled
        class MockMeta:
            def observe(self, *args, **kwargs):
                pass
        return MockMeta()
    
    global _meta_instance
    if _meta_instance is None:
        try:
            from meta_prime import MetaPrime
            _meta_instance = MetaPrime()
            _meta_instance.observe("shared_meta_initialized", {
                "timestamp": datetime.now().isoformat(),
                "purpose": "Prevent multiple MetaPrime instances"
            })
        except ImportError:
            # Fallback when meta system not available
            class MockMeta:
                def observe(self, *args, **kwargs):
                    pass
                    
            _meta_instance = MockMeta()
    
    return _meta_instance


def reset_shared_meta():
    """Reset the shared instance (for testing only)."""
    global _meta_instance
    _meta_instance = None