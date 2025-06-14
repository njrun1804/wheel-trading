"""Network and service configuration."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class NetworkConfig:
    """Network endpoints configuration."""
    
    # Service endpoints
    mlflow_tracking_uri: str = "http://localhost:5000"
    oauth_callback_host: str = "127.0.0.1"
    oauth_callback_port: int = 8182
    
    # Monitoring endpoints
    phoenix_port: int = 6006
    opik_port: int = 5173
    otlp_port: int = 4317
    otlp_endpoint: str = "http://127.0.0.1:4318"
    
    # Database defaults
    default_db_host: str = "localhost"
    default_db_port: int = 5432
    
    @property
    def oauth_redirect_uri(self) -> str:
        """Get OAuth redirect URI."""
        return f"http://{self.oauth_callback_host}:{self.oauth_callback_port}/callback"
    
    @property
    def service_endpoints(self) -> Dict[str, str]:
        """Get all service endpoints."""
        return {
            "mlflow": self.mlflow_tracking_uri,
            "oauth": self.oauth_redirect_uri,
            "phoenix": f"http://localhost:{self.phoenix_port}",
            "opik": f"http://localhost:{self.opik_port}",
            "otlp": self.otlp_endpoint,
        }


# Default instance
network_config = NetworkConfig()
