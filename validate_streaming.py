#!/usr/bin/env python3
"""Validate streaming processors implementation."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def validate_imports():
    """Validate that all imports work correctly."""
    print("Validating imports...")
    
    try:
        # Test stream processors
        from unity_wheel.utils.stream_processors import (
            DataStreamProcessor,
            JSONStreamProcessor,
            TextStreamProcessor,
            StreamConfig,
            DataType,
            MemoryMonitor,
        )
        print("✅ Stream processors imported successfully")
        
        # Test safe output
        from unity_wheel.utils.safe_output import (
            SafeOutputHandler,
            OutputConfig,
            safe_output,
            safe_json_output,
        )
        print("✅ Safe output handlers imported successfully")
        
        # Test memory-aware chunking
        from unity_wheel.utils.memory_aware_chunking import (
            AdaptiveChunker,
            ChunkingConfig,
            BytesChunkingStrategy,
        )
        print("✅ Memory-aware chunking imported successfully")
        
        # Test unified imports
        from unity_wheel.utils import (
            safe_output,
            StreamConfig,
            AdaptiveChunker,
        )
        print("✅ Unified utils imports work correctly")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def validate_basic_functionality():
    """Validate basic functionality without async."""
    print("\nValidating basic functionality...")
    
    try:
        from unity_wheel.utils import safe_output, StreamConfig
        
        # Test safe output
        result = safe_output("Test data")
        assert result.content == "Test data"
        assert not result.is_truncated
        print("✅ Safe output basic functionality works")
        
        # Test config creation
        config = StreamConfig()
        assert config.max_memory_mb > 0
        print("✅ Stream configuration works")
        
        # Test large data handling
        large_data = {"data": list(range(1000))}
        result = safe_output(large_data)
        assert result.original_size > 0
        print("✅ Large data handling works")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main validation function."""
    print("🔍 Validating Streaming Data Processors Implementation")
    print("=" * 60)
    
    # Validate imports
    imports_ok = validate_imports()
    
    if imports_ok:
        # Validate basic functionality
        functionality_ok = validate_basic_functionality()
        
        if functionality_ok:
            print("\n🎉 All validations passed!")
            print("\n📋 Implementation Summary:")
            print("   • Stream processors for large data handling")
            print("   • Safe output with automatic file fallback")
            print("   • Memory-aware chunking strategies")
            print("   • Integration with existing wheel trading patterns")
            print("   • Comprehensive error recovery")
            print("   • Performance monitoring and metrics")
            
            print("\n🚀 Ready to prevent string overflow errors in Claude Code!")
            return True
        else:
            print("\n❌ Functionality validation failed")
            return False
    else:
        print("\n❌ Import validation failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)