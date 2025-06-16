"""
Code-Specific Embedding Pipeline for Enhanced Semantic Search

This module provides improved embeddings specifically designed for code analysis:
1. AST-aware embeddings that understand code structure
2. Programming concept embeddings (async, inheritance, etc.)
3. Context-aware embeddings for better semantic matching
4. Optimized for coding queries with higher accuracy
"""

import ast
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CodeASTAnalyzer:
    """Analyzes Python code using AST to extract structural information."""

    def __init__(self):
        self.code_patterns = {
            "functions": [],
            "classes": [],
            "imports": [],
            "async_patterns": [],
            "inheritance": [],
            "exceptions": [],
            "decorators": [],
            "comprehensions": [],
        }

    def analyze_code(self, code: str, file_path: str) -> dict[str, Any]:
        """
        Analyze code structure using AST.

        Args:
            code: Python source code
            file_path: Path to the file being analyzed

        Returns:
            Dictionary containing structural analysis
        """
        analysis = {
            "functions": [],
            "classes": [],
            "imports": [],
            "async_functions": [],
            "inheritance_patterns": [],
            "exception_handlers": [],
            "decorators": [],
            "comprehensions": [],
            "complexity_indicators": [],
            "code_concepts": [],
        }

        try:
            tree = ast.parse(code)

            # Walk through AST nodes
            for node in ast.walk(tree):
                self._analyze_node(node, analysis)

            # Extract code concepts based on patterns
            analysis["code_concepts"] = self._extract_code_concepts(analysis)

            return analysis

        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e}")
            return analysis
        except Exception as e:
            logger.error(f"AST analysis failed for {file_path}: {e}")
            return analysis

    def _analyze_node(self, node: ast.AST, analysis: dict[str, Any]):
        """Analyze individual AST nodes."""

        # Function definitions
        if isinstance(node, ast.FunctionDef):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "is_async": False,
                "decorators": [
                    d.id if isinstance(d, ast.Name) else str(d)
                    for d in node.decorator_list
                ],
                "args": [arg.arg for arg in node.args.args],
            }
            analysis["functions"].append(func_info)

            # Check for decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Name):
                    analysis["decorators"].append(decorator.id)

        # Async function definitions
        elif isinstance(node, ast.AsyncFunctionDef):
            func_info = {
                "name": node.name,
                "line": node.lineno,
                "is_async": True,
                "decorators": [
                    d.id if isinstance(d, ast.Name) else str(d)
                    for d in node.decorator_list
                ],
                "args": [arg.arg for arg in node.args.args],
            }
            analysis["functions"].append(func_info)
            analysis["async_functions"].append(func_info)
            analysis["code_concepts"].append("async_programming")

        # Class definitions
        elif isinstance(node, ast.ClassDef):
            class_info = {
                "name": node.name,
                "line": node.lineno,
                "bases": [self._get_name(base) for base in node.bases],
                "decorators": [
                    d.id if isinstance(d, ast.Name) else str(d)
                    for d in node.decorator_list
                ],
            }
            analysis["classes"].append(class_info)

            # Check for inheritance
            if node.bases:
                for base in node.bases:
                    base_name = self._get_name(base)
                    analysis["inheritance_patterns"].append(
                        {"child": node.name, "parent": base_name, "line": node.lineno}
                    )
                analysis["code_concepts"].append("inheritance")

        # Import statements
        elif isinstance(node, ast.Import):
            for alias in node.names:
                import_info = {
                    "module": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                    "type": "import",
                }
                analysis["imports"].append(import_info)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                import_info = {
                    "module": f"{module}.{alias.name}" if module else alias.name,
                    "alias": alias.asname,
                    "line": node.lineno,
                    "type": "from_import",
                    "from_module": module,
                }
                analysis["imports"].append(import_info)

        # Exception handling
        elif isinstance(node, ast.Try):
            analysis["exception_handlers"].append(
                {
                    "line": node.lineno,
                    "handlers": len(node.handlers),
                    "has_finally": len(node.finalbody) > 0,
                    "has_else": len(node.orelse) > 0,
                }
            )
            analysis["code_concepts"].append("error_handling")

        # List/dict/set comprehensions
        elif isinstance(
            node, ast.ListComp | ast.DictComp | ast.SetComp | ast.GeneratorExp
        ):
            analysis["comprehensions"].append(
                {"type": type(node).__name__, "line": node.lineno}
            )
            analysis["code_concepts"].append("comprehensions")

        # Await expressions (async patterns)
        elif isinstance(node, ast.Await):
            analysis["code_concepts"].append("async_programming")

        # With statements (context managers)
        elif isinstance(node, ast.With):
            analysis["code_concepts"].append("context_management")

    def _get_name(self, node: ast.AST) -> str:
        """Extract name from AST node."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        else:
            return str(node)

    def _extract_code_concepts(self, analysis: dict[str, Any]) -> list[str]:
        """Extract high-level code concepts from analysis."""
        concepts = set(analysis.get("code_concepts", []))

        # Add concepts based on patterns
        if analysis["async_functions"]:
            concepts.add("async_programming")

        if analysis["inheritance_patterns"]:
            concepts.add("object_oriented_programming")
            concepts.add("inheritance")

        if analysis["exception_handlers"]:
            concepts.add("error_handling")

        if analysis["decorators"]:
            concepts.add("decorators")
            if any(
                dec in ["property", "staticmethod", "classmethod"]
                for dec in analysis["decorators"]
            ):
                concepts.add("advanced_python")

        if analysis["comprehensions"]:
            concepts.add("functional_programming")

        # Domain-specific concepts
        function_names = [f["name"].lower() for f in analysis["functions"]]
        if any("test" in name for name in function_names):
            concepts.add("testing")

        if any("async" in name for name in function_names):
            concepts.add("async_programming")

        if any(
            pattern in name
            for name in function_names
            for pattern in ["calculate", "compute", "math", "formula"]
        ):
            concepts.add("mathematical_computation")

        if any(
            pattern in name
            for name in function_names
            for pattern in ["trade", "wheel", "option", "delta", "gamma"]
        ):
            concepts.add("financial_trading")

        return list(concepts)


class CodeEmbeddingPipeline:
    """Enhanced embedding pipeline specifically for code analysis."""

    def __init__(self, embedding_dim: int = 1536):
        self.embedding_dim = embedding_dim
        self.ast_analyzer = CodeASTAnalyzer()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Code-specific weights for different components
        self.component_weights = {
            "content": 0.4,  # Raw content embedding
            "structure": 0.3,  # AST structure embedding
            "concepts": 0.2,  # Code concepts embedding
            "context": 0.1,  # File context embedding
        }

        # Common programming concepts for concept embeddings
        self.programming_concepts = {
            "async_programming": [0.1] * embedding_dim,
            "object_oriented_programming": [0.2] * embedding_dim,
            "functional_programming": [0.15] * embedding_dim,
            "error_handling": [0.18] * embedding_dim,
            "testing": [0.12] * embedding_dim,
            "mathematical_computation": [0.25] * embedding_dim,
            "financial_trading": [0.3] * embedding_dim,
            "data_processing": [0.22] * embedding_dim,
            "web_development": [0.17] * embedding_dim,
            "database_operations": [0.19] * embedding_dim,
        }

        # Initialize concept embeddings as random but consistent
        np.random.seed(42)  # For reproducible concept embeddings
        for concept in self.programming_concepts:
            self.programming_concepts[concept] = np.random.randn(embedding_dim).astype(
                np.float32
            )

        logger.info(
            f"✅ Code embedding pipeline initialized with {embedding_dim}D embeddings"
        )

    async def embed_code_content(
        self, code: str, file_path: str, context: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Generate code-aware embeddings for content.

        Args:
            code: Python source code
            file_path: Path to the file
            context: Additional context information

        Returns:
            Tuple of (embedding_vector, metadata)
        """

        # Analyze code structure
        ast_analysis = self.ast_analyzer.analyze_code(code, file_path)

        # Generate multiple embedding components
        embeddings = {}

        # 1. Content embedding (mock for now - in production use real embedding API)
        embeddings["content"] = await self._generate_content_embedding(code)

        # 2. Structure embedding based on AST
        embeddings["structure"] = self._generate_structure_embedding(ast_analysis)

        # 3. Concept embedding based on programming concepts
        embeddings["concepts"] = self._generate_concept_embedding(
            ast_analysis["code_concepts"]
        )

        # 4. Context embedding based on file information
        embeddings["context"] = self._generate_context_embedding(file_path, context)

        # Combine embeddings with weights
        combined_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        for component, weight in self.component_weights.items():
            if component in embeddings:
                combined_embedding += weight * embeddings[component]

        # Normalize
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm

        # Enhanced metadata
        metadata = {
            "file_path": file_path,
            "functions": [f["name"] for f in ast_analysis["functions"]],
            "classes": [c["name"] for c in ast_analysis["classes"]],
            "imports": [i["module"] for i in ast_analysis["imports"]],
            "async_functions": [f["name"] for f in ast_analysis["async_functions"]],
            "code_concepts": ast_analysis["code_concepts"],
            "inheritance_patterns": ast_analysis["inheritance_patterns"],
            "has_error_handling": len(ast_analysis["exception_handlers"]) > 0,
            "complexity_score": self._calculate_complexity(ast_analysis),
            "line_count": len(code.split("\n")),
            "keywords": self._extract_keywords(code, ast_analysis),
            "patterns": ast_analysis["code_concepts"],  # For pattern matching
        }

        return combined_embedding, metadata

    async def embed_code_query(self, query: str) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Generate embeddings for search queries with code-aware processing.

        Args:
            query: Search query

        Returns:
            Tuple of (query_embedding, query_metadata)
        """

        # Analyze query for coding intent
        query_analysis = self._analyze_query(query)

        # Generate query embedding components
        embeddings = {}

        # Content embedding for the query text
        embeddings["content"] = await self._generate_content_embedding(query)

        # Concept embedding based on detected concepts
        embeddings["concepts"] = self._generate_concept_embedding(
            query_analysis["concepts"]
        )

        # Intent embedding based on query type
        embeddings["intent"] = self._generate_intent_embedding(query_analysis["intent"])

        # Combine with query-specific weights
        query_weights = {"content": 0.5, "concepts": 0.3, "intent": 0.2}

        combined_embedding = np.zeros(self.embedding_dim, dtype=np.float32)
        for component, weight in query_weights.items():
            if component in embeddings:
                combined_embedding += weight * embeddings[component]

        # Normalize
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm

        query_metadata = {
            "original_query": query,
            "detected_concepts": query_analysis["concepts"],
            "query_intent": query_analysis["intent"],
            "query_type": query_analysis["type"],
            "expanded_terms": query_analysis["expanded_terms"],
        }

        return combined_embedding, query_metadata

    async def _generate_content_embedding(self, text: str) -> np.ndarray:
        """Generate content embedding (mock implementation)."""

        # In production, this would call an actual embedding API
        # For now, we create a consistent hash-based embedding

        # Simple hash-based embedding that's deterministic
        text_hash = hash(text.lower())
        np.random.seed(abs(text_hash) % (2**32))

        # Generate embedding with some text-aware features
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)

        # Add some basic text features
        word_count = len(text.split())
        char_count = len(text)

        # Encode basic text statistics in first few dimensions
        if len(embedding) > 10:
            embedding[0] = min(1.0, word_count / 100.0)  # Normalized word count
            embedding[1] = min(1.0, char_count / 1000.0)  # Normalized char count
            embedding[2] = 1.0 if "def " in text else 0.0  # Has function definition
            embedding[3] = 1.0 if "class " in text else 0.0  # Has class definition
            embedding[4] = 1.0 if "async " in text else 0.0  # Has async keyword
            embedding[5] = 1.0 if "import " in text else 0.0  # Has import
            embedding[6] = (
                1.0 if "try:" in text or "except" in text else 0.0
            )  # Has error handling

        return embedding

    def _generate_structure_embedding(self, ast_analysis: dict[str, Any]) -> np.ndarray:
        """Generate embedding based on code structure."""

        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Encode structural information
        if len(embedding) > 20:
            embedding[10] = min(1.0, len(ast_analysis["functions"]) / 10.0)
            embedding[11] = min(1.0, len(ast_analysis["classes"]) / 5.0)
            embedding[12] = min(1.0, len(ast_analysis["imports"]) / 20.0)
            embedding[13] = min(1.0, len(ast_analysis["async_functions"]) / 5.0)
            embedding[14] = min(1.0, len(ast_analysis["inheritance_patterns"]) / 3.0)
            embedding[15] = min(1.0, len(ast_analysis["exception_handlers"]) / 5.0)
            embedding[16] = min(1.0, len(ast_analysis["decorators"]) / 10.0)
            embedding[17] = min(1.0, len(ast_analysis["comprehensions"]) / 5.0)

        return embedding

    def _generate_concept_embedding(self, concepts: list[str]) -> np.ndarray:
        """Generate embedding based on programming concepts."""

        if not concepts:
            return np.zeros(self.embedding_dim, dtype=np.float32)

        # Combine concept embeddings
        combined = np.zeros(self.embedding_dim, dtype=np.float32)
        concept_count = 0

        for concept in concepts:
            if concept in self.programming_concepts:
                combined += self.programming_concepts[concept]
                concept_count += 1

        if concept_count > 0:
            combined = combined / concept_count

        return combined

    def _generate_context_embedding(
        self, file_path: str, context: dict[str, Any]
    ) -> np.ndarray:
        """Generate embedding based on file context."""

        embedding = np.zeros(self.embedding_dim, dtype=np.float32)

        # Encode file path information
        path_lower = file_path.lower()

        if len(embedding) > 30:
            embedding[20] = 1.0 if "test" in path_lower else 0.0
            embedding[21] = (
                1.0 if "util" in path_lower or "helper" in path_lower else 0.0
            )
            embedding[22] = 1.0 if "model" in path_lower else 0.0
            embedding[23] = 1.0 if "api" in path_lower else 0.0
            embedding[24] = 1.0 if "config" in path_lower else 0.0
            embedding[25] = (
                1.0 if "main" in path_lower or "__main__" in path_lower else 0.0
            )
            embedding[26] = 1.0 if "init" in path_lower else 0.0
            embedding[27] = (
                1.0
                if any(term in path_lower for term in ["trade", "wheel", "option"])
                else 0.0
            )

        return embedding

    def _generate_intent_embedding(self, intent: str) -> np.ndarray:
        """Generate embedding based on query intent."""

        intent_embeddings = {
            "find": np.random.randn(self.embedding_dim).astype(np.float32),
            "analyze": np.random.randn(self.embedding_dim).astype(np.float32),
            "list": np.random.randn(self.embedding_dim).astype(np.float32),
            "search": np.random.randn(self.embedding_dim).astype(np.float32),
            "identify": np.random.randn(self.embedding_dim).astype(np.float32),
        }

        # Seed with intent for consistency
        np.random.seed(hash(intent) % (2**32))

        return intent_embeddings.get(
            intent, np.zeros(self.embedding_dim, dtype=np.float32)
        )

    def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query to understand coding intent."""

        query_lower = query.lower()

        analysis = {
            "concepts": [],
            "intent": "search",
            "type": "generic",
            "expanded_terms": [],
        }

        # Detect programming concepts
        concept_patterns = {
            "async_programming": ["async", "await", "asynchronous"],
            "object_oriented_programming": ["class", "inheritance", "object"],
            "functional_programming": ["function", "lambda", "map", "filter"],
            "error_handling": ["exception", "error", "try", "catch", "except"],
            "testing": ["test", "unittest", "pytest"],
            "mathematical_computation": ["calculate", "compute", "math", "formula"],
            "financial_trading": [
                "trade",
                "wheel",
                "option",
                "delta",
                "gamma",
                "theta",
            ],
            "data_processing": ["data", "dataframe", "pandas", "process"],
        }

        for concept, patterns in concept_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                analysis["concepts"].append(concept)

        # Detect intent
        if any(word in query_lower for word in ["find", "locate", "search"]):
            analysis["intent"] = "find"
        elif any(word in query_lower for word in ["analyze", "examine", "review"]):
            analysis["intent"] = "analyze"
        elif any(word in query_lower for word in ["list", "show all", "display"]):
            analysis["intent"] = "list"
        elif any(word in query_lower for word in ["identify", "detect"]):
            analysis["intent"] = "identify"

        # Detect query type
        if "function" in query_lower or "def" in query_lower:
            analysis["type"] = "function_search"
        elif "class" in query_lower:
            analysis["type"] = "class_search"
        elif "import" in query_lower:
            analysis["type"] = "import_search"
        elif "async" in query_lower:
            analysis["type"] = "async_search"
        elif "error" in query_lower or "exception" in query_lower:
            analysis["type"] = "error_search"

        return analysis

    def _calculate_complexity(self, ast_analysis: dict[str, Any]) -> float:
        """Calculate code complexity score."""

        score = 0.0

        # Add complexity for different constructs
        score += len(ast_analysis["functions"]) * 1.0
        score += len(ast_analysis["classes"]) * 2.0
        score += len(ast_analysis["async_functions"]) * 1.5
        score += len(ast_analysis["inheritance_patterns"]) * 1.5
        score += len(ast_analysis["exception_handlers"]) * 1.2
        score += len(ast_analysis["decorators"]) * 0.5
        score += len(ast_analysis["comprehensions"]) * 0.8

        return score

    def _extract_keywords(self, code: str, ast_analysis: dict[str, Any]) -> list[str]:
        """Extract important keywords from code."""

        keywords = []

        # Add function names
        keywords.extend([f["name"] for f in ast_analysis["functions"]])

        # Add class names
        keywords.extend([c["name"] for c in ast_analysis["classes"]])

        # Add import modules
        keywords.extend([i["module"].split(".")[0] for i in ast_analysis["imports"]])

        # Add code concepts
        keywords.extend(ast_analysis["code_concepts"])

        # Add common programming keywords found in code
        programming_keywords = [
            "async",
            "await",
            "def",
            "class",
            "import",
            "try",
            "except",
            "with",
            "lambda",
            "yield",
            "return",
            "if",
            "else",
            "for",
            "while",
        ]

        code_lower = code.lower()
        for keyword in programming_keywords:
            if keyword in code_lower:
                keywords.append(keyword)

        return list(set(keywords))  # Remove duplicates

    def cleanup(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
