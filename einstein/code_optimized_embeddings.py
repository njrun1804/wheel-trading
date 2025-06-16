#!/usr/bin/env python3
"""
Code-Optimized Embedding System

High-quality embeddings specifically designed for programming code:
- Separates function definitions, class structures, docstrings
- Preserves syntax and semantic meaning
- Optimized chunking for code blocks
- Enhanced embeddings for programming concepts
- Pre-processes code for better semantic understanding

Performance targets:
- High-quality embeddings for code patterns
- Semantic understanding of programming concepts
- Optimal chunking that preserves code structure
- Enhanced search relevance for coding tasks
"""

import ast
import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""

    content: str
    chunk_type: str  # 'function', 'class', 'import', 'docstring', 'comment', 'mixed'
    start_line: int
    end_line: int
    name: str | None = None  # function/class name
    complexity_score: float = 0.0  # estimated complexity
    tokens: int = 0
    hash: str = ""
    keywords: list[str] = None
    imports: list[str] = None
    calls: list[str] = None


@dataclass
class CodeEmbeddingResult:
    """Result of code embedding with enhanced metadata."""

    embedding: np.ndarray
    chunk: CodeChunk
    semantic_features: dict[str, Any]
    quality_score: float


def extract_python_features(code: str) -> dict[str, Any]:
    """Extract semantic features from Python code."""
    features = {
        "functions": [],
        "classes": [],
        "imports": [],
        "decorators": [],
        "keywords": [],
        "complexity": 0,
        "docstring": None,
        "variables": [],
        "function_calls": [],
    }

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features["functions"].append(node.name)
                # Extract complexity based on nesting and statements
                features["complexity"] += len(node.body)
                if node.decorator_list:
                    features["decorators"].extend(
                        [
                            d.id if hasattr(d, "id") else str(d)
                            for d in node.decorator_list
                        ]
                    )

            elif isinstance(node, ast.ClassDef):
                features["classes"].append(node.name)
                if node.decorator_list:
                    features["decorators"].extend(
                        [
                            d.id if hasattr(d, "id") else str(d)
                            for d in node.decorator_list
                        ]
                    )

            elif isinstance(node, ast.Import):
                for alias in node.names:
                    features["imports"].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        features["imports"].append(f"{node.module}.{alias.name}")

            elif isinstance(node, ast.Call):
                if hasattr(node.func, "id"):
                    features["function_calls"].append(node.func.id)
                elif hasattr(node.func, "attr"):
                    features["function_calls"].append(node.func.attr)

            elif isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    features["variables"].append(node.id)

        # Extract docstring
        if (
            isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            features["docstring"] = tree.body[0].value.value

        # Extract keywords from code
        python_keywords = {
            "def",
            "class",
            "if",
            "else",
            "elif",
            "for",
            "while",
            "try",
            "except",
            "finally",
            "with",
            "import",
            "from",
            "return",
            "yield",
            "async",
            "await",
            "lambda",
            "global",
            "nonlocal",
            "assert",
            "del",
            "pass",
            "break",
            "continue",
        }

        words = re.findall(r"\b\w+\b", code.lower())
        features["keywords"] = [w for w in words if w in python_keywords]

    except SyntaxError:
        # If not valid Python, extract what we can with regex
        features["functions"] = re.findall(r"def\s+(\w+)", code)
        features["classes"] = re.findall(r"class\s+(\w+)", code)
        features["imports"] = re.findall(r"(?:import|from)\s+([\w.]+)", code)

    return features


def extract_programming_concepts(code: str, language: str = "python") -> list[str]:
    """Extract programming concepts and patterns from code."""
    concepts = []

    # Common programming patterns
    patterns = {
        "async_pattern": r"\basync\s+def\b|\bawait\b",
        "decorator_pattern": r"@\w+",
        "context_manager": r"\bwith\s+\w+",
        "exception_handling": r"\btry\b|\bexcept\b|\bfinally\b",
        "list_comprehension": r"\[.*for.*in.*\]",
        "dict_comprehension": r"\{.*for.*in.*\}",
        "generator": r"\byield\b",
        "property": r"@property",
        "static_method": r"@staticmethod",
        "class_method": r"@classmethod",
        "dataclass": r"@dataclass",
        "type_hints": r":\s*\w+[\[\]]*\s*=",
        "f_string": r'f["\'].*\{.*\}.*["\']',
        "logging": r"\blog\.|logger\.",
        "testing": r"def\s+test_|\bassert\b|pytest|unittest",
        "api_endpoint": r"@app\.|@router\.|@api\.",
        "database": r"\bselect\b|\binsert\b|\bupdate\b|\bdelete\b|\.query\(",
        "async_database": r"async\s+with.*\.begin\(\)|async\s+with.*\.transaction\(\)",
        "ml_framework": r"import\s+(?:torch|tensorflow|sklearn|numpy|pandas)",
        "web_framework": r"import\s+(?:flask|django|fastapi|starlette)",
    }

    for concept, pattern in patterns.items():
        if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
            concepts.append(concept)

    return concepts


def calculate_code_complexity(code: str) -> float:
    """Calculate complexity score for code chunk."""
    # Base complexity
    complexity = len(code.split("\n"))

    # Add complexity for control structures
    control_patterns = [
        r"\bif\b",
        r"\belif\b",
        r"\belse\b",
        r"\bfor\b",
        r"\bwhile\b",
        r"\btry\b",
        r"\bexcept\b",
        r"\bwith\b",
        r"\basync\s+def\b",
        r"\bdef\b",
        r"\bclass\b",
        r"\blambda\b",
    ]

    for pattern in control_patterns:
        complexity += len(re.findall(pattern, code)) * 2

    # Add complexity for nesting (rough approximation)
    lines = code.split("\n")
    max_indent = 0
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent // 4)  # Assuming 4-space indents

    complexity += max_indent * 3

    return min(complexity / 10.0, 10.0)  # Normalize to 0-10 scale


def intelligent_code_chunking(
    content: str, file_path: str, max_chunk_size: int = 1000
) -> list[CodeChunk]:
    """
    Intelligently chunk code preserving logical boundaries.

    Prioritizes:
    1. Function definitions
    2. Class definitions
    3. Import blocks
    4. Docstrings and comments
    5. Logical code blocks
    """
    chunks = []
    lines = content.split("\n")

    if not lines:
        return chunks

    # Try Python AST parsing first
    try:
        tree = ast.parse(content)
        return _chunk_python_ast(content, lines, tree, max_chunk_size)
    except SyntaxError:
        # Fall back to regex-based chunking
        return _chunk_with_regex(content, lines, file_path, max_chunk_size)


def _chunk_python_ast(
    content: str, lines: list[str], tree: ast.AST, max_chunk_size: int
) -> list[CodeChunk]:
    """Chunk Python code using AST analysis."""
    chunks = []
    processed_lines = set()

    # Extract top-level functions and classes
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef):
            start_line = node.lineno - 1  # AST uses 1-based line numbers
            end_line = (
                node.end_lineno - 1 if hasattr(node, "end_lineno") else start_line + 10
            )

            # Ensure we don't go out of bounds
            end_line = min(end_line, len(lines) - 1)

            if start_line < len(lines) and end_line < len(lines):
                chunk_content = "\n".join(lines[start_line : end_line + 1])

                chunk = CodeChunk(
                    content=chunk_content,
                    chunk_type="function"
                    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
                    else "class",
                    start_line=start_line + 1,  # Convert back to 1-based
                    end_line=end_line + 1,
                    name=node.name,
                    complexity_score=calculate_code_complexity(chunk_content),
                    tokens=len(chunk_content.split()),
                    hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
                )

                # Extract additional metadata
                features = extract_python_features(chunk_content)
                chunk.keywords = features.get("keywords", [])
                chunk.imports = features.get("imports", [])
                chunk.calls = features.get("function_calls", [])

                chunks.append(chunk)
                processed_lines.update(range(start_line, end_line + 1))

    # Handle imports block
    import_lines = []
    for i, line in enumerate(lines):
        if re.match(r"^\s*(import|from)\s+", line.strip()) and i not in processed_lines:
            import_lines.append((i, line))
            processed_lines.add(i)

    if import_lines:
        import_content = "\n".join(line for _, line in import_lines)
        chunks.append(
            CodeChunk(
                content=import_content,
                chunk_type="import",
                start_line=import_lines[0][0] + 1,
                end_line=import_lines[-1][0] + 1,
                tokens=len(import_content.split()),
                hash=hashlib.md5(import_content.encode()).hexdigest()[:8],
            )
        )

    # Handle remaining lines as mixed chunks
    remaining_lines = [
        i for i in range(len(lines)) if i not in processed_lines and lines[i].strip()
    ]

    if remaining_lines:
        current_chunk = []
        chunk_start = remaining_lines[0]

        for line_num in remaining_lines:
            current_chunk.append(lines[line_num])

            # Check if we should end this chunk
            if (
                len("\n".join(current_chunk)) > max_chunk_size
                or line_num == remaining_lines[-1]
            ):
                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        chunk_type="mixed",
                        start_line=chunk_start + 1,
                        end_line=line_num + 1,
                        complexity_score=calculate_code_complexity(chunk_content),
                        tokens=len(chunk_content.split()),
                        hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
                    )
                )

                current_chunk = []
                chunk_start = line_num + 1

    return sorted(chunks, key=lambda x: x.start_line)


def _chunk_with_regex(
    content: str, lines: list[str], file_path: str, max_chunk_size: int
) -> list[CodeChunk]:
    """Fallback chunking using regex patterns."""
    chunks = []
    current_chunk = []
    chunk_start = 1
    chunk_type = "mixed"

    for i, line in enumerate(lines):
        # Detect chunk boundaries
        if re.match(r"^\s*(def|class|async\s+def)\s+", line.strip()):
            # End current chunk if it exists
            if current_chunk:
                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        chunk_type=chunk_type,
                        start_line=chunk_start,
                        end_line=i,
                        complexity_score=calculate_code_complexity(chunk_content),
                        tokens=len(chunk_content.split()),
                        hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
                    )
                )

            # Start new chunk
            current_chunk = [line]
            chunk_start = i + 1
            chunk_type = "function" if "def" in line else "mixed"

        elif re.match(r"^\s*(import|from)\s+", line.strip()):
            if chunk_type != "import":
                # End current chunk
                if current_chunk:
                    chunk_content = "\n".join(current_chunk)
                    chunks.append(
                        CodeChunk(
                            content=chunk_content,
                            chunk_type=chunk_type,
                            start_line=chunk_start,
                            end_line=i,
                            complexity_score=calculate_code_complexity(chunk_content),
                            tokens=len(chunk_content.split()),
                            hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
                        )
                    )

                current_chunk = [line]
                chunk_start = i + 1
                chunk_type = "import"
            else:
                current_chunk.append(line)

        else:
            current_chunk.append(line)

            # Check size limit
            if len("\n".join(current_chunk)) > max_chunk_size:
                chunk_content = "\n".join(current_chunk)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        chunk_type=chunk_type,
                        start_line=chunk_start,
                        end_line=i + 1,
                        complexity_score=calculate_code_complexity(chunk_content),
                        tokens=len(chunk_content.split()),
                        hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
                    )
                )

                current_chunk = []
                chunk_start = i + 2
                chunk_type = "mixed"

    # Handle final chunk
    if current_chunk:
        chunk_content = "\n".join(current_chunk)
        chunks.append(
            CodeChunk(
                content=chunk_content,
                chunk_type=chunk_type,
                start_line=chunk_start,
                end_line=len(lines),
                complexity_score=calculate_code_complexity(chunk_content),
                tokens=len(chunk_content.split()),
                hash=hashlib.md5(chunk_content.encode()).hexdigest()[:8],
            )
        )

    return chunks


def enhance_code_embedding(
    code: str, base_embedding: np.ndarray, semantic_features: dict[str, Any]
) -> np.ndarray:
    """
    Enhance base embedding with code-specific features.

    Adds semantic information about:
    - Programming concepts
    - Code structure
    - Complexity
    - Keywords and patterns
    """
    if base_embedding is None or len(base_embedding) == 0:
        return base_embedding

    # Create feature vector for code-specific enhancements
    feature_vector = np.zeros(64)  # 64 additional dimensions for code features

    # Feature 0-9: Programming concepts (normalized)
    concepts = extract_programming_concepts(code)
    concept_mapping = {
        "async_pattern": 0,
        "decorator_pattern": 1,
        "context_manager": 2,
        "exception_handling": 3,
        "list_comprehension": 4,
        "generator": 5,
        "type_hints": 6,
        "testing": 7,
        "api_endpoint": 8,
        "database": 9,
    }

    for concept in concepts:
        if concept in concept_mapping:
            feature_vector[concept_mapping[concept]] = 1.0

    # Feature 10-19: Code structure
    if semantic_features:
        feature_vector[10] = min(len(semantic_features.get("functions", [])), 10) / 10.0
        feature_vector[11] = min(len(semantic_features.get("classes", [])), 10) / 10.0
        feature_vector[12] = min(len(semantic_features.get("imports", [])), 20) / 20.0
        feature_vector[13] = min(semantic_features.get("complexity", 0), 50) / 50.0
        feature_vector[14] = 1.0 if semantic_features.get("docstring") else 0.0
        feature_vector[15] = min(len(semantic_features.get("decorators", [])), 5) / 5.0
        feature_vector[16] = min(len(semantic_features.get("variables", [])), 20) / 20.0
        feature_vector[17] = (
            min(len(semantic_features.get("function_calls", [])), 20) / 20.0
        )

    # Feature 20-29: Code quality indicators
    feature_vector[20] = 1.0 if "TODO" in code.upper() else 0.0
    feature_vector[21] = 1.0 if "FIXME" in code.upper() else 0.0
    feature_vector[22] = (
        1.0 if re.search(r'""".*"""', code, re.DOTALL) else 0.0
    )  # Docstring
    feature_vector[23] = len(re.findall(r"#.*", code)) / max(
        len(code.split("\n")), 1
    )  # Comment ratio
    feature_vector[24] = 1.0 if "logger" in code or "logging" in code else 0.0
    feature_vector[25] = 1.0 if "test" in code.lower() or "assert" in code else 0.0

    # Feature 30-39: File type and context (can be extended)
    feature_vector[30] = 1.0 if "def __init__" in code else 0.0  # Constructor
    feature_vector[31] = (
        1.0 if "def __str__" in code or "def __repr__" in code else 0.0
    )  # String methods
    feature_vector[32] = (
        1.0 if "def __enter__" in code or "def __exit__" in code else 0.0
    )  # Context manager
    feature_vector[33] = (
        1.0 if re.search(r'if\s+__name__\s*==\s*["\']__main__["\']', code) else 0.0
    )  # Main guard

    # Feature 40-63: Reserved for future enhancements

    # Combine base embedding with enhanced features
    enhanced_embedding = np.concatenate([base_embedding, feature_vector])

    return enhanced_embedding


class CodeOptimizedEmbeddingSystem:
    """High-performance code-optimized embedding system."""

    def __init__(self, embedding_model: Any = None, max_chunk_size: int = 1000):
        """
        Initialize code-optimized embedding system.

        Args:
            embedding_model: Base embedding model (e.g., SentenceTransformer)
            max_chunk_size: Maximum size for code chunks
        """
        self.embedding_model = embedding_model
        self.max_chunk_size = max_chunk_size

        # MLX embedding function if no model provided
        if embedding_model is None:
            from einstein.mlx_embeddings import get_mlx_embedding_engine

            self.mlx_engine = get_mlx_embedding_engine(embed_dim=1536)
            self.embedding_function = self._mlx_embedding
        else:
            self.embedding_function = self._model_embedding

    def _mlx_embedding(self, text: str) -> tuple[np.ndarray, int]:
        """MLX-based embedding for production use."""
        try:
            embedding, token_count = self.mlx_engine.embed_text(text)
            return embedding, token_count
        except Exception as e:
            logger.error(f"MLX embedding failed: {e}")
            # Fallback to zero embedding
            return np.zeros(1536, dtype=np.float32), len(text.split())

    def _model_embedding(self, text: str) -> tuple[np.ndarray, int]:
        """Generate embedding using the provided model."""
        try:
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            token_count = len(text.split())  # Rough approximation
            return embedding.astype(np.float32), token_count
        except Exception as e:
            logger.error(f"Embedding model failed: {e}")
            # Fallback to MLX embedding
            return self._mlx_embedding(text)

    async def embed_file(
        self, file_path: str, language: str = "python"
    ) -> list[CodeEmbeddingResult]:
        """
        Generate optimized embeddings for a code file.

        Args:
            file_path: Path to the code file
            language: Programming language (currently supports 'python')

        Returns:
            List of CodeEmbeddingResult objects
        """
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return []

        if not content.strip():
            return []

        # Intelligently chunk the code
        chunks = intelligent_code_chunking(content, file_path, self.max_chunk_size)

        if not chunks:
            return []

        # Generate embeddings for each chunk
        results = []
        for chunk in chunks:
            try:
                # Extract semantic features
                semantic_features = (
                    extract_python_features(chunk.content)
                    if language == "python"
                    else {}
                )

                # Generate base embedding
                base_embedding, actual_tokens = self.embedding_function(chunk.content)
                chunk.tokens = actual_tokens

                # Enhance embedding with code-specific features
                enhanced_embedding = enhance_code_embedding(
                    chunk.content, base_embedding, semantic_features
                )

                # Calculate quality score
                quality_score = self._calculate_quality_score(chunk, semantic_features)

                result = CodeEmbeddingResult(
                    embedding=enhanced_embedding,
                    chunk=chunk,
                    semantic_features=semantic_features,
                    quality_score=quality_score,
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Failed to embed chunk from {file_path}: {e}")
                continue

        logger.info(f"Generated {len(results)} code embeddings for {file_path}")
        return results

    def _calculate_quality_score(
        self, chunk: CodeChunk, semantic_features: dict[str, Any]
    ) -> float:
        """Calculate quality score for an embedding."""
        score = 0.5  # Base score

        # Reward complete functions/classes
        if chunk.chunk_type in ["function", "class"] and chunk.name:
            score += 0.3

        # Reward docstrings
        if semantic_features.get("docstring"):
            score += 0.1

        # Reward type hints
        if "type_hints" in extract_programming_concepts(chunk.content):
            score += 0.1

        # Penalize very short or very long chunks
        if chunk.tokens < 10:
            score -= 0.2
        elif chunk.tokens > 2000:
            score -= 0.1

        # Reward meaningful content (not just imports or comments)
        if (
            chunk.chunk_type not in ["import"]
            and len(semantic_features.get("functions", [])) > 0
        ):
            score += 0.1

        return min(max(score, 0.0), 1.0)  # Clamp to 0-1 range

    def get_embedding_stats(self, results: list[CodeEmbeddingResult]) -> dict[str, Any]:
        """Get statistics about generated embeddings."""
        if not results:
            return {}

        chunk_types = {}
        total_tokens = 0
        quality_scores = []
        complexity_scores = []

        for result in results:
            chunk = result.chunk
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            total_tokens += chunk.tokens
            quality_scores.append(result.quality_score)
            complexity_scores.append(chunk.complexity_score)

        return {
            "total_embeddings": len(results),
            "chunk_types": chunk_types,
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / len(results),
            "avg_quality_score": np.mean(quality_scores),
            "avg_complexity_score": np.mean(complexity_scores),
            "embedding_dimension": len(results[0].embedding) if results else 0,
            "high_quality_chunks": sum(1 for score in quality_scores if score > 0.7),
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio

    async def test_code_embeddings():
        """Test the code-optimized embedding system."""
        # Initialize system
        system = CodeOptimizedEmbeddingSystem()

        # Test with current file
        test_file = __file__
        print(f"\nTesting with file: {test_file}")

        results = await system.embed_file(test_file)
        print(f"Generated {len(results)} embeddings")

        # Show statistics
        stats = system.get_embedding_stats(results)
        print("\nEmbedding Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Show sample chunks
        print("\nSample chunks:")
        for i, result in enumerate(results[:3]):  # Show first 3
            chunk = result.chunk
            print(f"\nChunk {i+1}:")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Lines: {chunk.start_line}-{chunk.end_line}")
            print(f"  Name: {chunk.name}")
            print(f"  Tokens: {chunk.tokens}")
            print(f"  Quality: {result.quality_score:.2f}")
            print(f"  Content preview: {chunk.content[:100]}...")

    # Run test
    asyncio.run(test_code_embeddings())
