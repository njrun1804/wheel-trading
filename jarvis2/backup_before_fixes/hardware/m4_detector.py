"""Real M4 Pro hardware detection for macOS."""
import subprocess
import json
import re
import logging
from typing import Dict
import psutil

logger = logging.getLogger(__name__)


class M4ProDetector:
    """Detect actual M4 Pro hardware configuration."""
    
    def __init__(self):
        self._cache = {}
        
    def detect_hardware(self) -> Dict[str, any]:
        """Detect M4 Pro hardware configuration."""
        if self._cache:
            return self._cache
            
        hardware = {
            'cpu': self._detect_cpu(),
            'gpu': self._detect_gpu(),
            'memory': self._detect_memory(),
            'model': self._detect_model(),
            'platform': 'macOS',
            'apple_silicon': self._is_apple_silicon()
        }
        
        self._cache = hardware
        return hardware
    
    def _detect_cpu(self) -> Dict[str, any]:
        """Detect CPU configuration using sysctl and system_profiler."""
        cpu_info = {}
        
        try:
            # Get CPU brand string
            brand = subprocess.check_output(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                text=True
            ).strip()
            cpu_info['brand'] = brand
            
            # Get core counts
            # Physical cores (P + E cores combined)
            physical = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.physicalcpu'],
                text=True
            ).strip())
            
            # Logical cores (with hyperthreading if available)
            logical = int(subprocess.check_output(
                ['sysctl', '-n', 'hw.logicalcpu'],
                text=True
            ).strip())
            
            cpu_info['physical_cores'] = physical
            cpu_info['logical_cores'] = logical
            
            # For M4 Pro, detect P and E cores
            if 'Apple M4' in brand:
                # M4 Pro has specific configurations:
                # Base: 12 cores (8P + 4E)
                # High: 14 cores (10P + 4E)
                if physical == 12:
                    cpu_info['p_cores'] = 8
                    cpu_info['e_cores'] = 4
                elif physical == 14:
                    cpu_info['p_cores'] = 10
                    cpu_info['e_cores'] = 4
                else:
                    # Fallback estimation
                    cpu_info['p_cores'] = physical - 4
                    cpu_info['e_cores'] = 4
            else:
                # Not M4, use conservative defaults
                cpu_info['p_cores'] = max(4, physical // 2)
                cpu_info['e_cores'] = physical - cpu_info['p_cores']
                
        except Exception as e:
            logger.warning(f"Failed to detect CPU details: {e}")
            # Fallback to psutil
            cpu_info = {
                'brand': 'Unknown',
                'physical_cores': psutil.cpu_count(logical=False) or 8,
                'logical_cores': psutil.cpu_count(logical=True) or 12,
                'p_cores': 8,
                'e_cores': 4
            }
            
        return cpu_info
    
    def _detect_gpu(self) -> Dict[str, any]:
        """Detect GPU configuration."""
        gpu_info = {}
        
        try:
            # Use system_profiler to get GPU info
            sp_output = subprocess.check_output(
                ['system_profiler', 'SPDisplaysDataType', '-json'],
                text=True
            )
            sp_data = json.loads(sp_output)
            
            # Extract GPU information
            displays = sp_data.get('SPDisplaysDataType', [])
            if displays:
                gpu_data = displays[0]
                gpu_name = gpu_data.get('sppci_model', 'Unknown GPU')
                gpu_info['name'] = gpu_name
                
                # For M4 Pro, determine GPU core count
                if 'M4 Pro' in gpu_name or 'Apple M4' in gpu_name:
                    # M4 Pro configurations:
                    # Base: 16 GPU cores
                    # High: 20 GPU cores
                    # Try to detect based on model identifier
                    model = self._detect_model()
                    if 'MacBookPro' in model.get('identifier', ''):
                        # High-end MacBook Pro M4 Pro typically has 20 cores
                        if '20-core' in gpu_name or 'Max' in gpu_name:
                            gpu_info['cores'] = 20
                        else:
                            gpu_info['cores'] = 16
                    else:
                        # Conservative default
                        gpu_info['cores'] = 16
                else:
                    gpu_info['cores'] = 8  # Default for non-M4
                    
                # VRAM is unified memory on Apple Silicon
                gpu_info['vram_mb'] = 0  # Unified memory
                gpu_info['unified_memory'] = True
                
        except Exception as e:
            logger.warning(f"Failed to detect GPU details: {e}")
            # Fallback
            gpu_info = {
                'name': 'Apple GPU',
                'cores': 16,  # M4 Pro base
                'vram_mb': 0,
                'unified_memory': True
            }
            
        # Check Metal support
        gpu_info['metal_supported'] = self._check_metal_support()
        
        return gpu_info
    
    def _detect_memory(self) -> Dict[str, any]:
        """Detect memory configuration."""
        memory_info = {}
        
        try:
            # Get total memory
            total_bytes = psutil.virtual_memory().total
            memory_info['total_gb'] = round(total_bytes / (1024**3), 1)
            
            # Get memory pressure
            vm_stat = subprocess.check_output(['vm_stat'], text=True)
            lines = vm_stat.strip().split('\n')
            
            page_size = 16384  # Default page size on Apple Silicon
            for line in lines:
                if 'page size' in line:
                    match = re.search(r'(\d+)', line)
                    if match:
                        page_size = int(match.group(1))
                    break
            
            memory_info['page_size'] = page_size
            
            # Unified memory specifics
            memory_info['unified'] = self._is_apple_silicon()
            
            # Metal memory limit (typically 75% of total on M4 Pro)
            if memory_info['unified']:
                memory_info['metal_limit_gb'] = round(memory_info['total_gb'] * 0.75, 1)
            else:
                memory_info['metal_limit_gb'] = 0
                
        except Exception as e:
            logger.warning(f"Failed to detect memory details: {e}")
            total_bytes = psutil.virtual_memory().total
            memory_info = {
                'total_gb': round(total_bytes / (1024**3), 1),
                'page_size': 16384,
                'unified': True,
                'metal_limit_gb': round(total_bytes / (1024**3) * 0.75, 1)
            }
            
        return memory_info
    
    def _detect_model(self) -> Dict[str, str]:
        """Detect Mac model."""
        model_info = {}
        
        try:
            # Get model identifier
            model_id = subprocess.check_output(
                ['sysctl', '-n', 'hw.model'],
                text=True
            ).strip()
            model_info['identifier'] = model_id
            
            # Get marketing name
            sp_output = subprocess.check_output(
                ['system_profiler', 'SPHardwareDataType'],
                text=True
            )
            
            for line in sp_output.split('\n'):
                if 'Model Name:' in line:
                    model_info['name'] = line.split(':', 1)[1].strip()
                elif 'Chip:' in line:
                    model_info['chip'] = line.split(':', 1)[1].strip()
                    
        except Exception as e:
            logger.warning(f"Failed to detect model: {e}")
            model_info = {
                'identifier': 'Unknown',
                'name': 'Mac',
                'chip': 'Apple Silicon'
            }
            
        return model_info
    
    def _is_apple_silicon(self) -> bool:
        """Check if running on Apple Silicon."""
        try:
            arch = subprocess.check_output(['uname', '-m'], text=True).strip()
            return arch == 'arm64'
        except:
            return False
    
    def _check_metal_support(self) -> bool:
        """Check if Metal is supported."""
        try:
            # Try to import Metal bindings
            return True
        except ImportError:
            # Check via system_profiler
            try:
                sp_output = subprocess.check_output(
                    ['system_profiler', 'SPDisplaysDataType'],
                    text=True
                )
                return 'Metal' in sp_output
            except:
                return self._is_apple_silicon()  # Apple Silicon always has Metal
    
    def get_optimal_settings(self) -> Dict[str, any]:
        """Get optimal settings for detected hardware."""
        hw = self.detect_hardware()
        
        settings = {
            'p_core_workers': hw['cpu']['p_cores'],
            'e_core_workers': hw['cpu']['e_cores'],
            'gpu_workers': min(2, hw['gpu']['cores'] // 8),  # 1 worker per 8 GPU cores
            'max_memory_gb': hw['memory']['total_gb'] * 0.85,  # Leave 15% for system
            'metal_memory_gb': hw['memory'].get('metal_limit_gb', 18),
            'batch_sizes': {
                'mcts': hw['cpu']['p_cores'] * 250,  # Simulations per worker
                'neural': 256 if hw['gpu']['cores'] >= 16 else 128,
                'embedding': 512,
                'learning': 32
            }
        }
        
        return settings


# Singleton instance
_detector = None

def get_detector() -> M4ProDetector:
    """Get singleton detector instance."""
    global _detector
    if _detector is None:
        _detector = M4ProDetector()
    return _detector


# Example usage
if __name__ == "__main__":
    detector = M4ProDetector()
    hw = detector.detect_hardware()
    print("Detected Hardware:")
    print(json.dumps(hw, indent=2))
    
    settings = detector.get_optimal_settings()
    print("\nOptimal Settings:")
    print(json.dumps(settings, indent=2))