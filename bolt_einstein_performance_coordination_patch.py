# Performance Coordination Patch for Bolt-Einstein Integration
# Add this to the RobustEinsteinIndexHub class in bolt/core/integration.py


class PerformanceCoordinatedEinsteinHub(RobustEinsteinIndexHub):
    """Einstein hub with performance coordination"""

    def __init__(self, project_root=None):
        super().__init__(project_root)
        self._performance_coordinator = None
        self._coordination_enabled = True

    def set_performance_coordinator(self, coordinator):
        """Set shared performance coordinator"""
        self._performance_coordinator = coordinator

    async def search(self, query, max_results=10, **kwargs):
        """Search with performance coordination"""

        if self._performance_coordinator and self._coordination_enabled:
            # Use coordinated search
            try:
                result = await self._performance_coordinator.coordinate_search(
                    query=query, system="einstein"
                )

                # If we got cached results, return them
                if result.get("cached", False):
                    return result.get("results", [])

                # Otherwise, perform the actual search
                return await super().search(query, max_results, **kwargs)

            except Exception as e:
                logger.warning(
                    f"Performance coordination failed: {e}, falling back to direct search"
                )
                return await super().search(query, max_results, **kwargs)
        else:
            # Direct search
            return await super().search(query, max_results, **kwargs)
