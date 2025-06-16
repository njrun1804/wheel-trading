"""
Unified Error Handling Interface

User-friendly interface for error handling across both Einstein and Bolt systems,
providing clear diagnostic information, recovery suggestions, and system status.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import error handling components
try:
    from bolt.error_handling.integration import get_integrated_error_manager, handle_any_error
    from bolt.error_handling.exceptions import BoltException
    from bolt.error_handling.graceful_degradation import get_degradation_manager
    BOLT_AVAILABLE = True
except ImportError:
    BOLT_AVAILABLE = False

try:
    from einstein.error_handling.diagnostics import get_einstein_diagnostics
    from einstein.error_handling.exceptions import EinsteinException
    EINSTEIN_AVAILABLE = True
except ImportError:
    EINSTEIN_AVAILABLE = False


class UserMessageLevel(Enum):
    """User message severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class UserFriendlyError:
    """User-friendly error representation."""
    title: str
    message: str
    level: UserMessageLevel
    component: str
    error_code: str
    timestamp: float
    suggestions: List[str]
    technical_details: Dict[str, Any]
    recovery_status: str
    system_impact: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            **asdict(self),
            'level': self.level.value,
            'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))
        }


@dataclass
class SystemStatus:
    """Overall system status information."""
    overall_health: str
    degradation_level: str
    active_errors: int
    recent_errors: int
    recovery_rate: float
    uptime_hours: float
    performance_status: str
    recommendations: List[str]
    component_status: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class ErrorHandlingInterface:
    """User-friendly interface for error handling and system diagnostics."""
    
    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path.cwd()
        self.logger = logging.getLogger(f"{__name__}.ErrorHandlingInterface")
        self.start_time = time.time()
        
        # Initialize components
        self.integrated_manager = None
        self.degradation_manager = None
        self.einstein_diagnostics = None
        
        if BOLT_AVAILABLE:
            try:
                self.integrated_manager = get_integrated_error_manager()
                self.degradation_manager = get_degradation_manager()
            except Exception as e:
                self.logger.warning(f"Failed to initialize Bolt components: {e}")
        
        if EINSTEIN_AVAILABLE:
            try:
                self.einstein_diagnostics = get_einstein_diagnostics(self.project_root)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Einstein diagnostics: {e}")
        
        # User message templates
        self.message_templates = {
            'search_failed': {
                'title': 'Search Unavailable',
                'message': 'The search function is temporarily unavailable. The system is working to restore it.',
                'suggestions': [
                    'Try your search again in a few moments',
                    'Use simpler search terms',
                    'Check if files exist in the project'
                ]
            },
            'index_corrupted': {
                'title': 'Search Index Issue',
                'message': 'The search index needs to be rebuilt to ensure accurate results.',
                'suggestions': [
                    'The system will automatically rebuild the index',
                    'This may take a few minutes',
                    'Search functionality will be limited during rebuild'
                ]
            },
            'memory_pressure': {
                'title': 'High Memory Usage',
                'message': 'The system is using more memory than optimal and has reduced some features.',
                'suggestions': [
                    'Close other applications to free memory',
                    'The system will automatically manage memory usage',
                    'Some advanced features may be temporarily disabled'
                ]
            },
            'gpu_failure': {
                'title': 'Graphics Processing Issue',
                'message': 'Graphics acceleration is unavailable. The system is using CPU processing instead.',
                'suggestions': [
                    'Performance may be slightly reduced',
                    'All functionality remains available',
                    'Consider restarting if issues persist'
                ]
            },
            'disk_space': {
                'title': 'Low Disk Space',
                'message': 'Available disk space is running low. Some features may be limited.',
                'suggestions': [
                    'Free up disk space by deleting unnecessary files',
                    'The system has reduced file caching',
                    'Consider archiving old project data'
                ]
            },
            'configuration_error': {
                'title': 'Configuration Issue',
                'message': 'There is a problem with the system configuration that needs attention.',
                'suggestions': [
                    'Check configuration files for syntax errors',
                    'Verify all required settings are present',
                    'Consider resetting to default configuration'
                ]
            },
            'dependency_missing': {
                'title': 'Missing Component',
                'message': 'A required software component is missing or not working properly.',
                'suggestions': [
                    'The system will try alternative methods',
                    'Some features may be unavailable',
                    'Check installation and requirements'
                ]
            },
            'network_issue': {
                'title': 'Network Connection Problem',
                'message': 'There is an issue with network connectivity affecting some features.',
                'suggestions': [
                    'Check your internet connection',
                    'The system will retry automatically',
                    'Local features remain available'
                ]
            },
            'system_overload': {
                'title': 'System Under Heavy Load',
                'message': 'The system is processing many operations and has temporarily reduced capacity.',
                'suggestions': [
                    'Operations may take longer than usual',
                    'The system will automatically balance the load',
                    'Consider reducing concurrent operations'
                ]
            }
        }
    
    async def handle_error_with_user_feedback(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_action: Optional[str] = None
    ) -> UserFriendlyError:
        """Handle an error and provide user-friendly feedback."""
        context = context or {}
        
        try:
            # Add user action context
            if user_action:
                context['user_action'] = user_action
            
            # Handle the error using integrated system
            recovery_success = False
            recovery_result = None
            
            if BOLT_AVAILABLE:
                recovery_success, recovery_result = await handle_any_error(error, context=context)
            
            # Generate user-friendly error
            user_error = self._generate_user_friendly_error(
                error, 
                context, 
                recovery_success, 
                recovery_result
            )
            
            # Log the user-friendly error
            self._log_user_error(user_error)
            
            return user_error
            
        except Exception as handling_error:
            self.logger.error(f"Error handling failed: {handling_error}", exc_info=True)
            
            # Return a generic user-friendly error
            return UserFriendlyError(
                title="Unexpected Issue",
                message="An unexpected issue occurred. The system is working to resolve it.",
                level=UserMessageLevel.ERROR,
                component="system",
                error_code="SYSTEM_ERROR",
                timestamp=time.time(),
                suggestions=[
                    "Try your action again in a moment",
                    "If the problem persists, restart the application",
                    "Check system resources and logs"
                ],
                technical_details={
                    'original_error': str(error),
                    'handling_error': str(handling_error)
                },
                recovery_status="unknown",
                system_impact="minimal"
            )
    
    def _generate_user_friendly_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_success: bool,
        recovery_result: Any
    ) -> UserFriendlyError:
        """Generate a user-friendly error message."""
        
        # Determine error characteristics
        error_str = str(error).lower()
        error_type = type(error).__name__
        
        # Select appropriate template
        template_key = self._select_message_template(error, error_str)
        template = self.message_templates.get(template_key, self.message_templates['system_overload'])
        
        # Determine message level
        if any(term in error_str for term in ['critical', 'fatal', 'crash']):
            level = UserMessageLevel.CRITICAL
        elif any(term in error_str for term in ['warning', 'degraded', 'limited']):
            level = UserMessageLevel.WARNING
        elif any(term in error_str for term in ['info', 'notice']):
            level = UserMessageLevel.INFO
        else:
            level = UserMessageLevel.ERROR
        
        # Determine component
        component = self._determine_component(error, error_str)
        
        # Generate error code
        error_code = getattr(error, 'error_code', f"{error_type.upper()}")
        
        # Customize message based on recovery
        message = template['message']
        suggestions = template['suggestions'].copy()
        
        if recovery_success:
            message += " The issue has been automatically resolved."
            suggestions.insert(0, "The system has recovered - you can continue normally")
            recovery_status = "recovered"
        else:
            message += " The system is still working on this issue."
            suggestions.append("If problems continue, manual intervention may be needed")
            recovery_status = "in_progress"
        
        # Determine system impact
        system_impact = self._assess_system_impact(error, recovery_success)
        
        # Collect technical details
        technical_details = {
            'error_type': error_type,
            'error_message': str(error),
            'recovery_success': recovery_success,
            'context': context
        }
        
        if hasattr(error, 'to_dict'):
            technical_details['error_details'] = error.to_dict()
        
        if recovery_result:
            technical_details['recovery_result'] = str(recovery_result)
        
        return UserFriendlyError(
            title=template['title'],
            message=message,
            level=level,
            component=component,
            error_code=error_code,
            timestamp=time.time(),
            suggestions=suggestions,
            technical_details=technical_details,
            recovery_status=recovery_status,
            system_impact=system_impact
        )
    
    def _select_message_template(self, error: Exception, error_str: str) -> str:
        """Select appropriate message template based on error type."""
        
        if 'search' in error_str or 'query' in error_str:
            return 'search_failed'
        elif 'index' in error_str or 'faiss' in error_str:
            return 'index_corrupted'
        elif 'memory' in error_str or isinstance(error, MemoryError):
            return 'memory_pressure'
        elif 'gpu' in error_str or 'metal' in error_str or 'cuda' in error_str:
            return 'gpu_failure'
        elif 'disk' in error_str or 'space' in error_str:
            return 'disk_space'
        elif 'config' in error_str or 'setting' in error_str:
            return 'configuration_error'
        elif 'import' in error_str or 'module' in error_str or isinstance(error, ImportError):
            return 'dependency_missing'
        elif 'network' in error_str or 'connection' in error_str:
            return 'network_issue'
        else:
            return 'system_overload'
    
    def _determine_component(self, error: Exception, error_str: str) -> str:
        """Determine which component the error relates to."""
        
        if isinstance(error, BoltException) if BOLT_AVAILABLE else 'bolt' in error_str:
            return 'bolt'
        elif isinstance(error, EinsteinException) if EINSTEIN_AVAILABLE else 'einstein' in error_str:
            return 'einstein'
        elif 'search' in error_str or 'index' in error_str:
            return 'search'
        elif 'embedding' in error_str or 'model' in error_str:
            return 'embedding'
        elif 'file' in error_str or 'watch' in error_str:
            return 'file_system'
        elif 'database' in error_str or 'db' in error_str:
            return 'database'
        elif 'gpu' in error_str or 'metal' in error_str:
            return 'hardware'
        elif 'network' in error_str or 'connection' in error_str:
            return 'network'
        else:
            return 'system'
    
    def _assess_system_impact(self, error: Exception, recovery_success: bool) -> str:
        """Assess the impact of the error on system functionality."""
        
        if recovery_success:
            return "resolved"
        
        error_str = str(error).lower()
        
        if any(term in error_str for term in ['critical', 'fatal', 'crash']):
            return "severe"
        elif any(term in error_str for term in ['index', 'search', 'database']):
            return "moderate"
        elif any(term in error_str for term in ['embedding', 'gpu', 'file']):
            return "limited"
        else:
            return "minimal"
    
    def _log_user_error(self, user_error: UserFriendlyError):
        """Log user-friendly error with appropriate level."""
        
        log_methods = {
            UserMessageLevel.INFO: self.logger.info,
            UserMessageLevel.WARNING: self.logger.warning,
            UserMessageLevel.ERROR: self.logger.error,
            UserMessageLevel.CRITICAL: self.logger.critical
        }
        
        log_method = log_methods.get(user_error.level, self.logger.error)
        
        log_method(
            f"User Error [{user_error.error_code}]: {user_error.title} - {user_error.message} "
            f"(Component: {user_error.component}, Impact: {user_error.system_impact})"
        )
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status for users."""
        
        try:
            current_time = time.time()
            uptime_hours = (current_time - self.start_time) / 3600
            
            # Initialize default values
            overall_health = "unknown"
            degradation_level = "normal"
            active_errors = 0
            recent_errors = 0
            recovery_rate = 100.0
            performance_status = "normal"
            recommendations = []
            component_status = {}
            
            # Get status from integrated manager if available
            if BOLT_AVAILABLE and self.integrated_manager:
                try:
                    health_summary = self.integrated_manager.get_system_health_summary()
                    
                    overall_health = health_summary.get('system_status', 'unknown')
                    degradation_level = health_summary.get('current_degradation_level', 'normal')
                    recent_errors = health_summary.get('total_errors_last_hour', 0)
                    recovery_rate = health_summary.get('recovery_success_rate', 100.0)
                    recommendations.extend(health_summary.get('recommendations', []))
                    
                    # Map degradation level to user-friendly terms
                    degradation_map = {
                        'normal': 'Normal Operation',
                        'reduced': 'Reduced Capacity',
                        'minimal': 'Minimal Operation',
                        'emergency': 'Emergency Mode'
                    }
                    degradation_level = degradation_map.get(degradation_level, degradation_level)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get Bolt system status: {e}")
            
            # Get Einstein diagnostics if available
            if EINSTEIN_AVAILABLE and self.einstein_diagnostics:
                try:
                    diagnostics = await self.einstein_diagnostics.run_diagnostics(quick=True)
                    component_status['einstein'] = diagnostics.overall_status.value
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get Einstein diagnostics: {e}")
                    component_status['einstein'] = 'unknown'
            
            # Determine performance status
            if degradation_level != 'Normal Operation':
                performance_status = "degraded"
            elif recent_errors > 10:
                performance_status = "stressed"
            elif recent_errors > 5:
                performance_status = "busy"
            else:
                performance_status = "optimal"
            
            # Add component status for Bolt
            if BOLT_AVAILABLE:
                component_status['bolt'] = 'operational' if overall_health in ['healthy', 'warning'] else 'degraded'
            
            # Generate user-friendly recommendations
            user_recommendations = self._generate_user_recommendations(
                overall_health, degradation_level, recent_errors, recovery_rate
            )
            recommendations.extend(user_recommendations)
            
            return SystemStatus(
                overall_health=overall_health,
                degradation_level=degradation_level,
                active_errors=active_errors,
                recent_errors=recent_errors,
                recovery_rate=recovery_rate,
                uptime_hours=uptime_hours,
                performance_status=performance_status,
                recommendations=list(set(recommendations)),  # Remove duplicates
                component_status=component_status
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            
            return SystemStatus(
                overall_health="unknown",
                degradation_level="unknown",
                active_errors=0,
                recent_errors=0,
                recovery_rate=0.0,
                uptime_hours=uptime_hours,
                performance_status="unknown",
                recommendations=["System status check failed - manual inspection recommended"],
                component_status={}
            )
    
    def _generate_user_recommendations(
        self,
        health: str,
        degradation: str,
        recent_errors: int,
        recovery_rate: float
    ) -> List[str]:
        """Generate user-friendly recommendations."""
        
        recommendations = []
        
        if degradation != 'Normal Operation':
            recommendations.append(f"System is in {degradation} - some features may be limited")
        
        if recent_errors > 10:
            recommendations.append("High error rate detected - consider restarting the application")
        elif recent_errors > 5:
            recommendations.append("Moderate error activity - monitor system performance")
        
        if recovery_rate < 50:
            recommendations.append("Low recovery success rate - manual intervention may be needed")
        elif recovery_rate < 80:
            recommendations.append("Some errors not recovering automatically - check system resources")
        
        if health == "emergency":
            recommendations.append("System in emergency mode - immediate attention required")
        elif health == "critical":
            recommendations.append("Critical system issues detected - check logs and restart if needed")
        elif health == "stressed":
            recommendations.append("System under stress - reduce workload or check resources")
        
        if not recommendations:
            recommendations.append("System operating normally - no action required")
        
        return recommendations
    
    def export_error_report(self, output_file: Optional[Path] = None) -> Path:
        """Export comprehensive error report for debugging."""
        
        if output_file is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            output_file = self.project_root / f"error_report_{timestamp}.json"
        
        try:
            report_data = {
                'timestamp': time.time(),
                'timestamp_str': time.strftime('%Y-%m-%d %H:%M:%S'),
                'project_root': str(self.project_root),
                'uptime_hours': (time.time() - self.start_time) / 3600,
                'components_available': {
                    'bolt': BOLT_AVAILABLE,
                    'einstein': EINSTEIN_AVAILABLE
                }
            }
            
            # Add system status
            try:
                import asyncio
                system_status = asyncio.get_event_loop().run_until_complete(self.get_system_status())
                report_data['system_status'] = system_status.to_dict()
            except Exception as e:
                report_data['system_status_error'] = str(e)
            
            # Add integrated manager data if available
            if BOLT_AVAILABLE and self.integrated_manager:
                try:
                    report_data['integrated_manager'] = self.integrated_manager.get_system_health_summary()
                except Exception as e:
                    report_data['integrated_manager_error'] = str(e)
            
            # Add Einstein diagnostics if available
            if EINSTEIN_AVAILABLE and self.einstein_diagnostics:
                try:
                    import asyncio
                    diagnostics = asyncio.get_event_loop().run_until_complete(
                        self.einstein_diagnostics.run_diagnostics()
                    )
                    report_data['einstein_diagnostics'] = diagnostics.to_dict()
                except Exception as e:
                    report_data['einstein_diagnostics_error'] = str(e)
            
            # Write report
            with open(output_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Error report exported to: {output_file}")
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to export error report: {e}")
            # Create minimal report
            minimal_report = {
                'timestamp': time.time(),
                'error': str(e),
                'message': 'Failed to generate full error report'
            }
            
            with open(output_file, 'w') as f:
                json.dump(minimal_report, f, indent=2)
            
            return output_file
    
    def get_user_help_text(self, error_code: Optional[str] = None) -> str:
        """Get user-friendly help text for errors or general troubleshooting."""
        
        if error_code:
            # Specific error help
            help_text = f"Help for Error {error_code}:\n\n"
            
            # Find matching template
            for template_key, template in self.message_templates.items():
                if error_code.lower() in template_key or template_key in error_code.lower():
                    help_text += f"Issue: {template['title']}\n"
                    help_text += f"Description: {template['message']}\n\n"
                    help_text += "Recommended Actions:\n"
                    for i, suggestion in enumerate(template['suggestions'], 1):
                        help_text += f"{i}. {suggestion}\n"
                    break
            else:
                help_text += "No specific help available for this error code.\n"
                help_text += "Please refer to the general troubleshooting guide below.\n"
        else:
            help_text = "General Troubleshooting Guide:\n\n"
        
        # General troubleshooting
        help_text += """
General Troubleshooting Steps:

1. Check System Resources:
   - Ensure sufficient memory (RAM) is available
   - Verify adequate disk space (at least 1GB free)
   - Monitor CPU usage (should be below 90%)

2. Restart Components:
   - Try restarting the application
   - Clear temporary files and caches
   - Restart file monitoring if search issues occur

3. Check Configuration:
   - Verify configuration files are valid
   - Ensure all required dependencies are installed
   - Check environment variables and settings

4. Index Maintenance:
   - Rebuild search indexes if search is slow
   - Clear and regenerate embeddings if needed
   - Update file indexes after major file changes

5. Performance Optimization:
   - Reduce concurrent operations if system is slow
   - Close unnecessary applications
   - Use degraded mode if full functionality isn't needed

6. Get Help:
   - Check application logs for detailed error messages
   - Export error report for technical support
   - Document steps to reproduce the issue

For persistent issues, export an error report and contact support.
"""
        
        return help_text


# Global interface instance
_error_interface: ErrorHandlingInterface | None = None


def get_error_handling_interface(project_root: Optional[Path] = None) -> ErrorHandlingInterface:
    """Get or create the global error handling interface."""
    global _error_interface
    if _error_interface is None:
        _error_interface = ErrorHandlingInterface(project_root)
    return _error_interface


# Convenience functions for easy integration

async def handle_user_error(
    error: Exception,
    user_action: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> UserFriendlyError:
    """Handle an error with user-friendly feedback."""
    interface = get_error_handling_interface()
    return await interface.handle_error_with_user_feedback(error, context, user_action)


async def get_user_system_status() -> SystemStatus:
    """Get user-friendly system status."""
    interface = get_error_handling_interface()
    return await interface.get_system_status()


def export_user_error_report(output_file: Optional[Path] = None) -> Path:
    """Export user-friendly error report."""
    interface = get_error_handling_interface()
    return interface.export_error_report(output_file)


def get_help_for_error(error_code: Optional[str] = None) -> str:
    """Get user help text for specific error or general troubleshooting."""
    interface = get_error_handling_interface()
    return interface.get_user_help_text(error_code)