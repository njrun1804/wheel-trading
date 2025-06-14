"""Code embeddings using lightweight methods optimized for M4 Pro."""
import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from dataclasses import dataclass
from .code_understanding import CodeAnalyzer


@dataclass
class CodeEmbedding:
    """Embedding for a piece of code."""
    code: str
    vector: np.ndarray
    metadata: Dict[str, Any]


class LightweightCodeEmbedder:
    """Lightweight code embedder using TF-IDF and code features."""
    
    def __init__(self, vector_dim: int = 128):
        self.vector_dim = vector_dim
        self.analyzer = CodeAnalyzer()
        
        # TF-IDF for code tokens
        self.tfidf = TfidfVectorizer(
            max_features=1000,
            token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b',  # Python identifiers
            stop_words=self._get_python_stop_words(),
            ngram_range=(1, 2)
        )
        
        # Dimensionality reduction
        self.svd = TruncatedSVD(n_components=vector_dim // 2)
        
        # Feature weights
        self.feature_weights = {
            'imports': 0.15,
            'functions': 0.25,
            'classes': 0.20,
            'complexity': 0.10,
            'patterns': 0.15,
            'identifiers': 0.15
        }
        
        self.is_fitted = False
    
    def fit(self, code_samples: List[str]):
        """Fit the embedder on code samples."""
        if not code_samples:
            return
        
        # Extract text features
        texts = []
        for code in code_samples:
            text = self._extract_text_features(code)
            texts.append(text)
        
        # Fit TF-IDF
        tfidf_matrix = self.tfidf.fit_transform(texts)
        
        # Fit SVD
        self.svd.fit(tfidf_matrix)
        
        self.is_fitted = True
    
    def embed(self, code: str) -> np.ndarray:
        """Generate embedding for code."""
        # Extract structural features
        structural_features = self._extract_structural_features(code)
        
        # Extract text features
        text = self._extract_text_features(code)
        
        if self.is_fitted:
            # Use trained TF-IDF
            tfidf_vec = self.tfidf.transform([text])
            text_embedding = self.svd.transform(tfidf_vec)[0]
        else:
            # Fallback: simple hash-based embedding
            text_embedding = self._hash_embedding(text, self.vector_dim // 2)
        
        # Combine embeddings
        embedding = np.concatenate([structural_features, text_embedding])
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)
    
    def _extract_structural_features(self, code: str) -> np.ndarray:
        """Extract structural features from code."""
        features = np.zeros(self.vector_dim // 2, dtype=np.float32)
        
        try:
            context = self.analyzer.analyze(code)
            
            # Basic counts (normalized)
            features[0] = min(len(context.imports) / 10, 1.0)
            features[1] = min(len(context.functions) / 5, 1.0)
            features[2] = min(len(context.classes) / 3, 1.0)
            features[3] = min(context.complexity / 20, 1.0)
            
            # Import patterns
            import_types = self._categorize_imports(context.imports)
            for i, (category, count) in enumerate(import_types.items()):
                if i + 4 < len(features):
                    features[i + 4] = min(count / 5, 1.0)
            
            # Function patterns
            func_patterns = self._analyze_function_patterns(context.functions)
            offset = 10
            for i, (pattern, score) in enumerate(func_patterns.items()):
                if offset + i < len(features):
                    features[offset + i] = score
            
            # Code patterns
            pattern_scores = self._detect_code_patterns(code)
            offset = 20
            for i, score in enumerate(pattern_scores):
                if offset + i < len(features):
                    features[offset + i] = score
            
        except:
            # If parsing fails, use basic metrics
            features[0] = len(code.split('\n')) / 100  # Lines
            features[1] = len(code.split()) / 500  # Words
            features[2] = code.count('def ') / 10  # Functions
            features[3] = code.count('class ') / 5  # Classes
        
        return features
    
    def _extract_text_features(self, code: str) -> str:
        """Extract text representation from code."""
        try:
            context = self.analyzer.analyze(code)
            
            # Combine various text elements
            text_parts = []
            
            # Function and class names
            for func in context.functions:
                text_parts.append(func['name'])
                if func['docstring']:
                    text_parts.append(func['docstring'])
            
            for cls in context.classes:
                text_parts.append(cls['name'])
                text_parts.extend(cls['methods'])
            
            # Variable names
            text_parts.extend(context.variables)
            
            # Import names
            for imp in context.imports:
                text_parts.append(imp.replace('.', ' '))
            
            # Comments and docstrings
            for line in code.split('\n'):
                if line.strip().startswith('#'):
                    text_parts.append(line.strip('# '))
            
            return ' '.join(text_parts)
        
        except:
            # Fallback: extract identifiers
            import re
            identifiers = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', code)
            return ' '.join(identifiers)
    
    def _categorize_imports(self, imports: List[str]) -> Dict[str, int]:
        """Categorize imports by type."""
        categories = {
            'stdlib': 0,
            'data': 0,
            'web': 0,
            'ml': 0,
            'testing': 0,
            'other': 0
        }
        
        stdlib_modules = {'os', 'sys', 'math', 'random', 'json', 'datetime', 
                         'collections', 'itertools', 'functools'}
        data_modules = {'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn'}
        web_modules = {'flask', 'django', 'requests', 'fastapi', 'aiohttp'}
        ml_modules = {'sklearn', 'torch', 'tensorflow', 'keras', 'mlx'}
        test_modules = {'pytest', 'unittest', 'mock'}
        
        for imp in imports:
            base_module = imp.split('.')[0]
            
            if base_module in stdlib_modules:
                categories['stdlib'] += 1
            elif base_module in data_modules:
                categories['data'] += 1
            elif base_module in web_modules:
                categories['web'] += 1
            elif base_module in ml_modules:
                categories['ml'] += 1
            elif base_module in test_modules:
                categories['testing'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _analyze_function_patterns(self, functions: List[Dict]) -> Dict[str, float]:
        """Analyze function patterns."""
        patterns = {
            'has_return': 0.0,
            'has_params': 0.0,
            'has_docstring': 0.0,
            'is_async': 0.0,
            'has_decorators': 0.0,
            'avg_params': 0.0
        }
        
        if not functions:
            return patterns
        
        total_params = 0
        for func in functions:
            if func.get('returns'):
                patterns['has_return'] += 1
            if func.get('args', []):
                patterns['has_params'] += 1
                total_params += len(func['args'])
            if func.get('docstring'):
                patterns['has_docstring'] += 1
            if func.get('is_async'):
                patterns['is_async'] += 1
            if func.get('decorators'):
                patterns['has_decorators'] += 1
        
        # Normalize
        n = len(functions)
        for key in patterns:
            if key != 'avg_params':
                patterns[key] /= n
        
        patterns['avg_params'] = total_params / n if n > 0 else 0
        patterns['avg_params'] = min(patterns['avg_params'] / 5, 1.0)  # Normalize
        
        return patterns
    
    def _detect_code_patterns(self, code: str) -> List[float]:
        """Detect common code patterns."""
        patterns = []
        
        # List comprehensions
        patterns.append(1.0 if '[' in code and 'for' in code and ']' in code else 0.0)
        
        # Generator expressions
        patterns.append(1.0 if '(' in code and 'for' in code and ')' in code else 0.0)
        
        # Context managers
        patterns.append(1.0 if 'with ' in code else 0.0)
        
        # Exception handling
        patterns.append(min(code.count('try:') / 3, 1.0))
        
        # Decorators
        patterns.append(min(code.count('@') / 5, 1.0))
        
        # Type hints
        patterns.append(1.0 if '->' in code or ': ' in code else 0.0)
        
        # Async/await
        patterns.append(1.0 if 'async ' in code or 'await ' in code else 0.0)
        
        # Class inheritance
        patterns.append(1.0 if 'class ' in code and '(' in code else 0.0)
        
        return patterns
    
    def _hash_embedding(self, text: str, dim: int) -> np.ndarray:
        """Simple hash-based embedding for fallback."""
        # Use multiple hash functions
        vec = np.zeros(dim, dtype=np.float32)
        
        words = text.split()
        for word in words:
            # Multiple hash projections
            for i in range(3):
                h = hash(word + str(i)) % dim
                vec[h] += 1.0
        
        # Normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        
        return vec
    
    def _get_python_stop_words(self) -> List[str]:
        """Get Python-specific stop words."""
        return [
            'self', 'def', 'class', 'import', 'from', 'return', 'if', 'else',
            'elif', 'for', 'while', 'in', 'is', 'not', 'and', 'or', 'as',
            'pass', 'break', 'continue', 'True', 'False', 'None'
        ]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate similarity between embeddings."""
        return float(cosine_similarity([embedding1], [embedding2])[0, 0])
    
    def save(self, path: Path):
        """Save embedder state."""
        state = {
            'vector_dim': self.vector_dim,
            'is_fitted': self.is_fitted,
            'feature_weights': self.feature_weights
        }
        
        if self.is_fitted:
            state['tfidf'] = self.tfidf
            state['svd'] = self.svd
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, path: Path):
        """Load embedder state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.vector_dim = state['vector_dim']
        self.is_fitted = state['is_fitted']
        self.feature_weights = state['feature_weights']
        
        if self.is_fitted:
            self.tfidf = state['tfidf']
            self.svd = state['svd']


class MLXCodeEmbedder:
    """Code embedder using MLX for M4 Pro optimization."""
    
    def __init__(self, model_name: str = "mlx-community/bert-base-uncased"):
        """Initialize with MLX model."""
        try:
            import mlx.core as mx
            import mlx.nn as nn
            from mlx_lm import load
            
            self.mx = mx
            self.nn = nn
            self.model, self.tokenizer = load(model_name)
            self.use_mlx = True
        except:
            # Fallback to lightweight
            self.use_mlx = False
            self.fallback = LightweightCodeEmbedder()
    
    def embed(self, code: str) -> np.ndarray:
        """Generate embedding using MLX."""
        if not self.use_mlx:
            return self.fallback.embed(code)
        
        # Tokenize
        tokens = self.tokenizer.encode(code, max_length=512, truncation=True)
        
        # Convert to MLX array
        input_ids = self.mx.array(tokens).reshape(1, -1)
        
        # Get embeddings
        with self.mx.no_grad():
            outputs = self.model(input_ids)
            # Use CLS token embedding
            embedding = outputs[0, 0, :].numpy()
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.astype(np.float32)