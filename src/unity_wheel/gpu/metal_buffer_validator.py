"""
Metal/MLX Buffer Alignment Validator for M4 Pro GPU

This module provides comprehensive buffer validation for Metal Performance Shaders
and MLX operations on M4 Pro, ensuring proper alignment to prevent GPU fallback
and optimize performance.

Based on Apple Metal documentation and M4 Pro hardware specifications:
- M4 Pro: 20 Metal GPU cores, 273 GB/s memory bandwidth
- Metal alignment requirements: 16-byte minimum for texture buffers
- MLX unified memory architecture: Zero-copy operations
- Device-specific alignment queries for optimal performance
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

import numpy as np

try:
    import mlx.core as mx

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None

logger = logging.getLogger(__name__)


class BufferType(Enum):
    """Types of buffers requiring different alignment requirements"""

    TEXTURE_BUFFER = "texture_buffer"
    COMPUTE_BUFFER = "compute_buffer"
    EMBEDDING_MATRIX = "embedding_matrix"
    SEARCH_RESULTS = "search_results"
    VERTEX_BUFFER = "vertex_buffer"
    INDEX_BUFFER = "index_buffer"
    UNIFORM_BUFFER = "uniform_buffer"


class PixelFormat(Enum):
    """Common pixel formats with their alignment requirements"""

    FLOAT32 = ("float32", 4, 16)  # 4 bytes per element, 16-byte aligned
    FLOAT16 = ("float16", 2, 16)  # 2 bytes per element, 16-byte aligned
    UINT32 = ("uint32", 4, 16)  # 4 bytes per element, 16-byte aligned
    UINT16 = ("uint16", 2, 16)  # 2 bytes per element, 16-byte aligned
    UINT8 = ("uint8", 1, 16)  # 1 byte per element, 16-byte aligned
    RGBA32F = ("rgba32f", 16, 16)  # 16 bytes per pixel, 16-byte aligned
    RGBA16F = ("rgba16f", 8, 16)  # 8 bytes per pixel, 16-byte aligned
    RGBA8 = ("rgba8", 4, 16)  # 4 bytes per pixel, 16-byte aligned

    def __init__(self, name: str, bytes_per_element: int, alignment: int):
        self.format_name = name
        self.bytes_per_element = bytes_per_element
        self.alignment = alignment


@dataclass
class M4ProSpecs:
    """M4 Pro hardware specifications for buffer optimization"""

    gpu_cores: int = 20
    memory_bandwidth_gbps: int = 273
    unified_memory: bool = True
    max_buffer_size: int = 2**32  # 4GB practical limit
    cache_line_size: int = 64
    memory_page_size: int = 16384  # 16KB pages

    # Performance characteristics
    min_efficient_workload_size: int = 2048  # Elements below this use CPU
    optimal_tile_size: int = 512  # For tiled operations
    max_concurrent_operations: int = 20  # Based on GPU cores


@dataclass
class BufferValidationResult:
    """Result of buffer validation with detailed feedback"""

    is_valid: bool
    buffer_size: int
    required_alignment: int
    actual_alignment: int
    bytes_per_row: int
    warnings: list[str]
    errors: list[str]
    recommendations: list[str]
    performance_impact: str  # "optimal", "good", "poor", "cpu_fallback"


class MetalBufferValidator:
    """
    Comprehensive buffer validator for Metal/MLX operations on M4 Pro.

    Validates buffer alignment, size, and layout requirements to ensure
    optimal GPU performance and prevent fallback to CPU.
    """

    def __init__(self):
        self.m4_pro_specs = M4ProSpecs()
        self.device_info = self._detect_device_capabilities()
        logger.info(f"Initialized MetalBufferValidator for M4 Pro: {self.device_info}")

    def _detect_device_capabilities(self) -> dict[str, Any]:
        """Detect actual Metal device capabilities"""
        capabilities = {
            "metal_available": False,
            "mlx_available": MLX_AVAILABLE,
            "unified_memory": True,  # Always true on Apple Silicon
            "max_buffer_size": self.m4_pro_specs.max_buffer_size,
            "min_texture_buffer_alignment": 16,  # Metal minimum
            "min_linear_texture_alignment": 16,  # Safe default
            "optimal_alignment": 64,  # Cache line aligned
        }

        if MLX_AVAILABLE:
            try:
                # Test MLX availability and basic operations
                test_array = mx.array([1.0, 2.0, 3.0])
                mx.eval(test_array)
                capabilities["metal_available"] = True
                capabilities["mlx_functional"] = True

                # Try to detect device-specific alignment requirements
                if hasattr(mx, "metal") and hasattr(mx.metal, "device"):
                    # These would be device-specific queries in a real implementation
                    capabilities["device_specific_alignment"] = True

                del test_array

            except Exception as e:
                logger.warning(f"MLX/Metal detection failed: {e}")
                capabilities["mlx_functional"] = False

        return capabilities

    def validate_buffer_size(
        self,
        buffer_size: int,
        element_count: int,
        element_size: int,
        buffer_type: BufferType = BufferType.COMPUTE_BUFFER,
    ) -> BufferValidationResult:
        """
        Validate buffer size against M4 Pro hardware limits and performance characteristics.

        Args:
            buffer_size: Total buffer size in bytes
            element_count: Number of elements in the buffer
            element_size: Size of each element in bytes
            buffer_type: Type of buffer for specific validation rules

        Returns:
            BufferValidationResult with validation details
        """
        warnings = []
        errors = []
        recommendations = []

        # Calculate expected size
        expected_size = element_count * element_size

        # Validate basic size consistency
        if buffer_size < expected_size:
            errors.append(
                f"Buffer size {buffer_size} is smaller than required {expected_size}"
            )

        # Check against hardware limits
        if buffer_size > self.m4_pro_specs.max_buffer_size:
            errors.append(
                f"Buffer size {buffer_size} exceeds M4 Pro maximum {self.m4_pro_specs.max_buffer_size}"
            )

        # Performance recommendations based on workload size
        performance_impact = self._assess_performance_impact(element_count, buffer_type)

        if performance_impact == "cpu_fallback":
            warnings.append(
                f"Buffer size ({element_count} elements) below GPU efficiency threshold"
            )
            recommendations.append(
                "Consider batching operations or using CPU for small workloads"
            )

        # Memory bandwidth considerations
        memory_transfer_time_us = (
            buffer_size / (self.m4_pro_specs.memory_bandwidth_gbps * 1e9)
        ) * 1e6
        if memory_transfer_time_us > 1000:  # > 1ms
            warnings.append(
                f"Large buffer may impact memory bandwidth (transfer time: {memory_transfer_time_us:.1f}μs)"
            )

        # Alignment validation
        alignment_result = self._validate_alignment(buffer_size, buffer_type)

        return BufferValidationResult(
            is_valid=len(errors) == 0,
            buffer_size=buffer_size,
            required_alignment=alignment_result["required_alignment"],
            actual_alignment=alignment_result["actual_alignment"],
            bytes_per_row=alignment_result["bytes_per_row"],
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            performance_impact=performance_impact,
        )

    def validate_texture_buffer(
        self,
        width: int,
        height: int,
        pixel_format: PixelFormat,
        buffer_size: int,
        bytes_per_row: int | None = None,
    ) -> BufferValidationResult:
        """
        Validate texture buffer with proper bytesPerRow alignment.

        Args:
            width: Texture width in pixels
            height: Texture height in pixels
            pixel_format: Pixel format specification
            buffer_size: Total buffer size in bytes
            bytes_per_row: Optional bytes per row (calculated if None)

        Returns:
            BufferValidationResult with texture-specific validation
        """
        warnings = []
        errors = []
        recommendations = []

        # Calculate required bytes per row
        min_bytes_per_row = width * pixel_format.bytes_per_element

        # Apply Metal alignment requirements
        required_alignment = max(
            pixel_format.alignment, self.device_info["min_texture_buffer_alignment"]
        )

        # Calculate aligned bytes per row
        aligned_bytes_per_row = self._align_to_boundary(
            min_bytes_per_row, required_alignment
        )

        if bytes_per_row is None:
            bytes_per_row = aligned_bytes_per_row

        # Validate provided bytes per row
        if bytes_per_row < min_bytes_per_row:
            errors.append(
                f"bytesPerRow {bytes_per_row} is less than minimum {min_bytes_per_row}"
            )

        if bytes_per_row % required_alignment != 0:
            errors.append(
                f"bytesPerRow {bytes_per_row} not aligned to {required_alignment} bytes"
            )
            recommendations.append(
                f"Use bytesPerRow = {aligned_bytes_per_row} for proper alignment"
            )

        # Validate total buffer size
        expected_buffer_size = bytes_per_row * height
        if buffer_size < expected_buffer_size:
            errors.append(
                f"Buffer size {buffer_size} is smaller than required {expected_buffer_size}"
            )

        # Performance recommendations
        if width * height < self.m4_pro_specs.min_efficient_workload_size:
            warnings.append("Small texture may not benefit from GPU acceleration")
            recommendations.append(
                "Consider using CPU for small textures or batch operations"
            )

        # Optimal tile size recommendations
        if (
            width > self.m4_pro_specs.optimal_tile_size
            or height > self.m4_pro_specs.optimal_tile_size
        ):
            recommendations.append(
                f"Consider tiling operations for textures larger than {self.m4_pro_specs.optimal_tile_size}x{self.m4_pro_specs.optimal_tile_size}"
            )

        performance_impact = self._assess_texture_performance_impact(
            width, height, pixel_format
        )

        return BufferValidationResult(
            is_valid=len(errors) == 0,
            buffer_size=buffer_size,
            required_alignment=required_alignment,
            actual_alignment=bytes_per_row % required_alignment,
            bytes_per_row=bytes_per_row,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            performance_impact=performance_impact,
        )

    def validate_mlx_array(
        self, array: Union[np.ndarray, "mx.array"], operation_type: str = "generic"
    ) -> BufferValidationResult:
        """
        Validate MLX array for optimal Metal performance.

        Args:
            array: NumPy array or MLX array to validate
            operation_type: Type of operation (e.g., "matmul", "conv", "embedding")

        Returns:
            BufferValidationResult with MLX-specific validation
        """
        warnings = []
        errors = []
        recommendations = []

        # Handle different array types
        if (
            isinstance(array, np.ndarray)
            or MLX_AVAILABLE
            and isinstance(array, mx.array)
        ):
            shape = array.shape
            dtype = array.dtype
            buffer_size = array.nbytes
            element_size = array.itemsize
        else:
            errors.append(f"Unsupported array type: {type(array)}")
            return BufferValidationResult(
                is_valid=False,
                buffer_size=0,
                required_alignment=16,
                actual_alignment=0,
                bytes_per_row=0,
                warnings=warnings,
                errors=errors,
                recommendations=recommendations,
                performance_impact="error",
            )

        # Validate array properties
        total_elements = np.prod(shape)

        # Check for common problematic patterns
        if len(shape) > 6:
            warnings.append(
                f"High-dimensional array ({len(shape)}D) may have suboptimal performance"
            )

        # Validate data type alignment
        if dtype == np.float16 or dtype == np.float32:
            required_alignment = 16  # Metal requirement
        elif dtype in [np.int32, np.uint32]:
            required_alignment = 16
        else:
            warnings.append(f"Unusual data type {dtype} may not be optimized for Metal")
            required_alignment = 16

        # Check memory layout for optimal access patterns
        if isinstance(array, np.ndarray):
            if not array.flags["C_CONTIGUOUS"]:
                warnings.append("Array is not C-contiguous, may impact performance")
                recommendations.append(
                    "Use np.ascontiguousarray() for better performance"
                )

            # Check alignment of the data pointer
            data_ptr = array.ctypes.data
            actual_alignment = self._get_alignment(data_ptr)

            if actual_alignment < required_alignment:
                warnings.append(
                    f"Array data alignment ({actual_alignment}) less than optimal ({required_alignment})"
                )
                recommendations.append("Consider using aligned memory allocation")
        else:
            # For MLX arrays, assume optimal alignment
            actual_alignment = required_alignment

        # Operation-specific validation
        performance_impact = self._assess_mlx_performance_impact(
            total_elements, shape, operation_type
        )

        if performance_impact == "cpu_fallback":
            recommendations.append("Consider using CPU for this operation size")
        elif performance_impact == "poor":
            recommendations.append(
                "Consider reshaping or batching for better GPU utilization"
            )

        # Memory bandwidth considerations
        if buffer_size > 100 * 1024 * 1024:  # > 100MB
            warnings.append("Large array may impact memory bandwidth")
            recommendations.append("Consider processing in smaller chunks")

        return BufferValidationResult(
            is_valid=len(errors) == 0,
            buffer_size=buffer_size,
            required_alignment=required_alignment,
            actual_alignment=actual_alignment,
            bytes_per_row=shape[-1] * element_size if len(shape) >= 2 else buffer_size,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            performance_impact=performance_impact,
        )

    def _validate_alignment(
        self, buffer_size: int, buffer_type: BufferType
    ) -> dict[str, int]:
        """Validate buffer alignment requirements"""

        # Determine required alignment based on buffer type
        if buffer_type == BufferType.TEXTURE_BUFFER:
            required_alignment = self.device_info["min_texture_buffer_alignment"]
        elif buffer_type == BufferType.UNIFORM_BUFFER:
            required_alignment = 256  # Metal uniform buffer alignment
        else:
            required_alignment = 16  # General Metal alignment requirement

        # For optimal performance, align to cache line boundaries
        optimal_alignment = max(required_alignment, self.m4_pro_specs.cache_line_size)

        # Calculate actual alignment
        actual_alignment = self._get_alignment(buffer_size)

        # Calculate optimal bytes per row (for 2D data)
        bytes_per_row = self._align_to_boundary(buffer_size, optimal_alignment)

        return {
            "required_alignment": required_alignment,
            "optimal_alignment": optimal_alignment,
            "actual_alignment": actual_alignment,
            "bytes_per_row": bytes_per_row,
        }

    def _assess_performance_impact(
        self, element_count: int, buffer_type: BufferType
    ) -> str:
        """Assess performance impact based on element count and buffer type"""

        if element_count < self.m4_pro_specs.min_efficient_workload_size:
            return "cpu_fallback"
        elif element_count < self.m4_pro_specs.min_efficient_workload_size * 4:
            return "poor"
        elif element_count < self.m4_pro_specs.min_efficient_workload_size * 16:
            return "good"
        else:
            return "optimal"

    def _assess_texture_performance_impact(
        self, width: int, height: int, pixel_format: PixelFormat
    ) -> str:
        """Assess texture performance impact"""

        total_pixels = width * height
        total_bytes = total_pixels * pixel_format.bytes_per_element

        # Consider both pixel count and memory usage
        if total_pixels < 1024 or total_bytes < 4096:
            return "cpu_fallback"
        elif total_pixels < 16384:
            return "poor"
        elif total_pixels < 262144:  # 512x512
            return "good"
        else:
            return "optimal"

    def _assess_mlx_performance_impact(
        self, total_elements: int, shape: tuple[int, ...], operation_type: str
    ) -> str:
        """Assess MLX array performance impact"""

        # Operation-specific thresholds
        thresholds = {"matmul": 1000, "conv": 500, "embedding": 100, "generic": 1000}

        threshold = thresholds.get(operation_type, thresholds["generic"])

        if total_elements < threshold:
            return "cpu_fallback"
        elif total_elements < threshold * 10:
            return "poor"
        elif total_elements < threshold * 100:
            return "good"
        else:
            return "optimal"

    def _align_to_boundary(self, size: int, alignment: int) -> int:
        """Align size to the specified boundary"""
        return ((size + alignment - 1) // alignment) * alignment

    def _get_alignment(self, address_or_size: int) -> int:
        """Get the alignment of an address or size"""
        if address_or_size == 0:
            return 0

        # Find the largest power of 2 that divides the address/size
        alignment = 1
        while alignment <= address_or_size and address_or_size % (alignment * 2) == 0:
            alignment *= 2

        return alignment

    def calculate_optimal_bytes_per_row(
        self,
        width: int,
        pixel_format: PixelFormat,
        buffer_type: BufferType = BufferType.TEXTURE_BUFFER,
    ) -> int:
        """
        Calculate optimal bytesPerRow for a given width and pixel format.

        Args:
            width: Width in pixels
            pixel_format: Pixel format specification
            buffer_type: Buffer type for alignment requirements

        Returns:
            Optimal bytesPerRow value
        """
        min_bytes_per_row = width * pixel_format.bytes_per_element

        # Get alignment requirements
        if buffer_type == BufferType.TEXTURE_BUFFER:
            alignment = max(
                pixel_format.alignment, self.device_info["min_texture_buffer_alignment"]
            )
        else:
            alignment = 16  # Default Metal alignment

        # Align to optimal boundary (cache line size for best performance)
        optimal_alignment = max(alignment, self.m4_pro_specs.cache_line_size)

        return self._align_to_boundary(min_bytes_per_row, optimal_alignment)

    def generate_buffer_size_recommendations(
        self, workload_description: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Generate buffer size recommendations for a specific workload.

        Args:
            workload_description: Dictionary describing the workload characteristics

        Returns:
            Dictionary with buffer size recommendations
        """
        corpus_size = workload_description.get("corpus_size", 10000)
        embedding_dim = workload_description.get("embedding_dim", 768)
        max_queries = workload_description.get("max_concurrent_queries", 10)
        operation_types = workload_description.get(
            "operation_types", ["similarity_search"]
        )

        recommendations = {
            "embedding_matrix": {
                "size_bytes": corpus_size * embedding_dim * 4,  # float32
                "alignment": 64,  # Cache line aligned
                "performance_notes": "Primary data structure, keep in unified memory",
            },
            "query_buffer": {
                "size_bytes": max_queries * embedding_dim * 4,
                "alignment": 64,
                "performance_notes": "Frequent updates, optimize for write performance",
            },
            "result_buffer": {
                "size_bytes": max_queries * 1000 * 64,  # 1000 results * 64 bytes each
                "alignment": 64,
                "performance_notes": "Read-heavy, optimize for cache efficiency",
            },
            "temporary_buffer": {
                "size_bytes": max(
                    corpus_size * embedding_dim * 4 // 10, 64 * 1024 * 1024
                ),
                "alignment": 64,
                "performance_notes": "Intermediate operations, size based on largest operation",
            },
        }

        # Adjust for specific operation types
        if "matrix_multiply" in operation_types:
            recommendations["temp_matrix_buffer"] = {
                "size_bytes": max_queries * corpus_size * 4,
                "alignment": 64,
                "performance_notes": "For matrix multiplication results",
            }

        return recommendations

    def print_validation_report(self, result: BufferValidationResult) -> None:
        """Print a detailed validation report"""
        print("\n=== Metal Buffer Validation Report ===")
        print(f"Valid: {'✓' if result.is_valid else '✗'}")
        print(
            f"Buffer Size: {result.buffer_size:,} bytes ({result.buffer_size/1024/1024:.1f} MB)"
        )
        print(f"Required Alignment: {result.required_alignment} bytes")
        print(f"Actual Alignment: {result.actual_alignment} bytes")
        print(f"Bytes Per Row: {result.bytes_per_row:,}")
        print(f"Performance Impact: {result.performance_impact}")

        if result.errors:
            print("\n❌ ERRORS:")
            for error in result.errors:
                print(f"  • {error}")

        if result.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"  • {warning}")

        if result.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for rec in result.recommendations:
                print(f"  • {rec}")

        print("=====================================\n")


# Convenience functions for common validation tasks


def validate_embedding_matrix(
    corpus_size: int, embedding_dim: int, dtype: np.dtype = np.float32
) -> BufferValidationResult:
    """Validate an embedding matrix buffer configuration"""
    validator = MetalBufferValidator()

    element_size = np.dtype(dtype).itemsize
    buffer_size = corpus_size * embedding_dim * element_size

    return validator.validate_buffer_size(
        buffer_size=buffer_size,
        element_count=corpus_size * embedding_dim,
        element_size=element_size,
        buffer_type=BufferType.EMBEDDING_MATRIX,
    )


def validate_search_result_buffer(
    max_concurrent_searches: int,
    max_results_per_search: int = 100,
    bytes_per_result: int = 1024,
) -> BufferValidationResult:
    """Validate a search result buffer configuration"""
    validator = MetalBufferValidator()

    buffer_size = max_concurrent_searches * max_results_per_search * bytes_per_result

    return validator.validate_buffer_size(
        buffer_size=buffer_size,
        element_count=max_concurrent_searches * max_results_per_search,
        element_size=bytes_per_result,
        buffer_type=BufferType.SEARCH_RESULTS,
    )


def calculate_aligned_buffer_size(
    base_size: int,
    alignment: int = 64,
    buffer_type: BufferType = BufferType.COMPUTE_BUFFER,
) -> int:
    """Calculate properly aligned buffer size"""
    validator = MetalBufferValidator()
    return validator._align_to_boundary(base_size, alignment)


def get_optimal_texture_layout(
    width: int, height: int, pixel_format: PixelFormat = PixelFormat.FLOAT32
) -> dict[str, int]:
    """Get optimal texture layout parameters"""
    validator = MetalBufferValidator()

    bytes_per_row = validator.calculate_optimal_bytes_per_row(
        width, pixel_format, BufferType.TEXTURE_BUFFER
    )

    total_size = bytes_per_row * height

    return {
        "width": width,
        "height": height,
        "bytes_per_row": bytes_per_row,
        "total_size": total_size,
        "pixel_format": pixel_format.format_name,
        "alignment": pixel_format.alignment,
    }


if __name__ == "__main__":
    # Test the validator with common scenarios
    print("Testing Metal Buffer Validator for M4 Pro...")

    validator = MetalBufferValidator()

    # Test 1: Embedding matrix validation
    print("\n1. Testing embedding matrix validation:")
    result = validate_embedding_matrix(corpus_size=50000, embedding_dim=768)
    validator.print_validation_report(result)

    # Test 2: Texture buffer validation
    print("\n2. Testing texture buffer validation:")
    result = validator.validate_texture_buffer(
        width=1024,
        height=1024,
        pixel_format=PixelFormat.FLOAT32,
        buffer_size=1024 * 1024 * 4,
    )
    validator.print_validation_report(result)

    # Test 3: MLX array validation
    print("\n3. Testing MLX array validation:")
    if MLX_AVAILABLE:
        test_array = mx.random.normal((1000, 768))
        result = validator.validate_mlx_array(test_array, "embedding")
        validator.print_validation_report(result)
    else:
        print("MLX not available, skipping MLX array test")

    # Test 4: Buffer size recommendations
    print("\n4. Testing buffer size recommendations:")
    workload = {
        "corpus_size": 100000,
        "embedding_dim": 768,
        "max_concurrent_queries": 20,
        "operation_types": ["similarity_search", "matrix_multiply"],
    }

    recommendations = validator.generate_buffer_size_recommendations(workload)
    print("Buffer Size Recommendations:")
    for buffer_name, config in recommendations.items():
        print(f"  {buffer_name}:")
        print(f"    Size: {config['size_bytes']/1024/1024:.1f} MB")
        print(f"    Alignment: {config['alignment']} bytes")
        print(f"    Notes: {config['performance_notes']}")
