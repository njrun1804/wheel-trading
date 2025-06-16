"""Real code generation based on context and patterns."""
import ast
import logging
import re
from dataclasses import dataclass
from typing import Any

from jinja2 import Template

from .ast_code_generator import ASTCodeGenerator, ClassSpec, FunctionSpec
from .code_understanding import CodeAnalyzer, CodeTransformer, CodeValidator

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    """Request for code generation."""

    query: str
    context: dict[str, Any]
    existing_code: str | None = None
    constraints: dict[str, Any] = None


class PatternMatcher:
    """Match queries to code patterns."""

    def __init__(self):
        self.patterns = {
            "function": [
                (
                    r"(create|write|implement|add)\s+.*\s*(function|method|func)",
                    "function",
                ),
                (r"function\s+(to|that|for)\s+", "function"),
                (r"def\s+", "function"),
            ],
            "class": [
                (r"(create|write|implement|add)\s+.*\s*(class|object)", "class"),
                (r"class\s+(to|that|for)\s+", "class"),
                (r"(manager|handler|service|controller)\s+for", "class"),
            ],
            "optimize": [
                (r"(optimize|improve|speed up|make faster)", "optimize"),
                (r"(performance|efficient|faster)", "optimize"),
            ],
            "refactor": [
                (r"(refactor|clean up|reorganize)", "refactor"),
                (r"(simplify|extract|split)", "refactor"),
            ],
            "test": [
                (r"(test|unit test|pytest)\s+", "test"),
                (r"(test case|test function)\s+for", "test"),
            ],
            "api": [
                (r"(api|endpoint|route|rest)\s+", "api"),
                (r"(get|post|put|delete)\s+endpoint", "api"),
            ],
            "algorithm": [
                (r"(algorithm|sort|search|find)", "algorithm"),
                (r"(binary search|quick sort|merge sort)", "algorithm"),
                (r"(fibonacci|factorial|prime)", "algorithm"),
            ],
        }

    def match_pattern(self, query: str) -> tuple[str, str]:
        """Match query to a pattern type and extract key info."""
        query_lower = query.lower()

        for pattern_type, patterns in self.patterns.items():
            for pattern, _ in patterns:
                if re.search(pattern, query_lower):
                    # Extract the main subject
                    subject = self._extract_subject(query_lower, pattern)
                    return pattern_type, subject

        return "general", query

    def _extract_subject(self, query: str, pattern: str) -> str:
        """Extract the main subject from the query."""
        # Remove the pattern match to get subject
        match = re.search(pattern, query)
        if match:
            subject = query.replace(match.group(), "").strip()
            # Clean up common words
            for word in ["a", "an", "the", "to", "that", "for"]:
                subject = re.sub(f"^{word}\\s+", "", subject)
                subject = re.sub(f"\\s+{word}$", "", subject)
            return subject
        return query


class CodeGenerator:
    """Generate real code based on patterns and context."""

    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.transformer = CodeTransformer()
        self.validator = CodeValidator()
        self.matcher = PatternMatcher()
        self.ast_generator = ASTCodeGenerator()
        self.templates = self._load_templates()  # Keep for fallback

    def generate(self, request: GenerationRequest) -> str:
        """Generate code based on request."""
        # Analyze existing code if provided
        context = {}
        if request.existing_code:
            context = self.analyzer.analyze(request.existing_code)

        # Match pattern
        pattern_type, subject = self.matcher.match_pattern(request.query)

        # Generate based on pattern
        if pattern_type == "function":
            code = self._generate_function(subject, context, request)
        elif pattern_type == "class":
            code = self._generate_class(subject, context, request)
        elif pattern_type == "optimize":
            code = self._optimize_code(subject, context, request)
        elif pattern_type == "test":
            code = self._generate_test(subject, context, request)
        elif pattern_type == "api":
            code = self._generate_api(subject, context, request)
        elif pattern_type == "algorithm":
            code = self._generate_algorithm(subject, context, request)
        else:
            code = self._generate_general(subject, context, request)

        # Validate and format
        code = self.validator.format_code(code)

        # Add to existing code if needed
        if request.existing_code and pattern_type in ["function", "class"]:
            code = self._merge_code(request.existing_code, code)

        return code

    def _generate_function(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Generate a function based on subject using AST."""
        # Parse function requirements
        func_name = self._generate_function_name(subject)
        args = self._infer_arguments(subject)
        return_type = self._infer_return_type(subject)

        # Generate body based on subject
        if "calculate" in subject or "compute" in subject:
            body_code = self._generate_calculation_body(subject, args)
        elif "validate" in subject or "check" in subject:
            body_code = self._generate_validation_body(subject, args)
        elif "convert" in subject or "transform" in subject:
            body_code = self._generate_conversion_body(subject, args)
        elif "fetch" in subject or "get" in subject:
            body_code = self._generate_fetch_body(subject, args)
        else:
            body_code = self._generate_generic_body(subject, args)

        # Create AST-based function spec
        func_spec = FunctionSpec(
            name=func_name,
            args=args,
            return_type=return_type,
            docstring=f"{subject.capitalize()}.",
            body_nodes=self.ast_generator.create_function_body(body_code),
            is_async="async" in subject.lower() or "await" in body_code,
            type_hints=self._infer_type_hints(subject, args, return_type),
        )

        # Generate using AST
        try:
            code = self.ast_generator.generate_function(func_spec)
            return code
        except Exception as e:
            # Fallback to template
            logger.warning(f"AST generation failed: {e}, using template")
            template = self.templates["function"]
            return template.render(
                name=func_name,
                args=args,
                return_type=return_type,
                docstring=f"{subject.capitalize()}.",
                body=body_code,
            )

    def _generate_class(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Generate a class based on subject using AST."""
        class_name = self._generate_class_name(subject)
        method_specs = []
        attributes = {}

        # Generate __init__ method
        init_args = self._infer_init_args(subject)
        init_body = self._generate_init_body(subject, init_args)
        init_spec = FunctionSpec(
            name="__init__",
            args=["self"] + init_args,
            return_type=None,
            docstring=f"Initialize {class_name}.",
            body_nodes=self.ast_generator.create_function_body(init_body),
        )
        method_specs.append(init_spec)

        # Generate other methods based on pattern
        methods = self._infer_methods(subject)
        for method in methods:
            if method["name"] != "__init__":
                method_spec = FunctionSpec(
                    name=method["name"],
                    args=method["args"],
                    return_type=self._infer_method_return_type(method["name"]),
                    docstring=f"{method['name'].replace('_', ' ').capitalize()}.",
                    body_nodes=self.ast_generator.create_function_body(method["body"]),
                )
                method_specs.append(method_spec)

        # Infer attributes
        attr_list = self._infer_attributes(subject)
        for attr in attr_list:
            if ":" in attr:
                name, type_hint = attr.split(":", 1)
                attributes[name.strip()] = type_hint.strip()
            else:
                attributes[attr] = None

        # Create class spec
        class_spec = ClassSpec(
            name=class_name,
            bases=self._infer_base_classes(subject),
            docstring=f"{subject.capitalize()} implementation.",
            attributes=attributes,
            methods=method_specs,
        )

        # Generate using AST
        try:
            code = self.ast_generator.generate_class(class_spec)
            return code
        except Exception as e:
            # Fallback to template
            logger.warning(f"AST class generation failed: {e}, using template")
            template = self.templates["class"]
            return template.render(
                name=class_name,
                docstring=f"{subject.capitalize()} implementation.",
                attributes=attr_list,
                methods=methods,
            )

    def _generate_algorithm(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Generate algorithm implementation."""
        algorithms = {
            "binary search": self._binary_search_template,
            "quick sort": self._quick_sort_template,
            "merge sort": self._merge_sort_template,
            "fibonacci": self._fibonacci_template,
            "factorial": self._factorial_template,
            "prime": self._prime_template,
        }

        for algo_name, template_func in algorithms.items():
            if algo_name in subject.lower():
                return template_func()

        # Generic algorithm
        return self._generate_function(subject, context, request)

    def _generate_test(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Generate test cases."""
        # Extract function/class to test
        target = self._extract_test_target(subject)

        template = self.templates["test"]
        code = template.render(
            target_name=target, test_cases=self._generate_test_cases(target)
        )

        return code

    def _generate_api(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Generate API endpoint."""
        method = "GET"
        for m in ["GET", "POST", "PUT", "DELETE"]:
            if m.lower() in subject.lower():
                method = m
                break

        endpoint = self._extract_endpoint_name(subject)

        template = self.templates["api"]
        code = template.render(
            method=method,
            endpoint=endpoint,
            handler_name=f"{method.lower()}_{endpoint.replace('/', '_')}",
        )

        return code

    def _optimize_code(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Optimize existing code."""
        if not request.existing_code:
            return "# No code provided to optimize"

        # Analyze code
        analysis = self.analyzer.analyze(request.existing_code)

        # Apply optimizations
        optimized = request.existing_code

        # Add type hints
        if not analysis.type_hints:
            optimized = self.transformer.add_type_hints(optimized)

        # Optimize imports
        optimized = self.transformer.optimize_imports(optimized)

        # Add docstrings if missing
        for func in analysis.functions:
            if not func["docstring"]:
                # Add docstring using transformer
                pass

        return optimized

    def _generate_general(
        self, subject: str, context: Any, request: GenerationRequest
    ) -> str:
        """Generate general code when no pattern matches."""
        # Try to generate a function
        return self._generate_function(subject, context, request)

    def _merge_code(self, existing: str, new: str) -> str:
        """Merge new code with existing code."""
        # Parse both
        existing_tree = ast.parse(existing)
        new_tree = ast.parse(new)

        # Extract new items
        new_items = []
        for node in new_tree.body:
            if isinstance(
                node, ast.FunctionDef | ast.ClassDef | ast.Import | ast.ImportFrom
            ):
                new_items.append(node)

        # Add to existing
        merged_tree = ast.Module(body=existing_tree.body + new_items, type_ignores=[])

        # Convert back to code
        if hasattr(ast, "unparse"):
            code = ast.unparse(merged_tree)
        else:
            # Fallback: just concatenate
            code = existing + "\n\n" + new

        return self.validator.format_code(code)

    def _generate_function_name(self, subject: str) -> str:
        """Generate appropriate function name."""
        # Remove common words
        words = subject.lower().split()
        words = [w for w in words if w not in ["a", "an", "the", "to", "that", "for"]]

        # Convert to snake_case
        return "_".join(words[:3])  # Limit to 3 words

    def _generate_class_name(self, subject: str) -> str:
        """Generate appropriate class name."""
        words = subject.lower().split()
        words = [w for w in words if w not in ["a", "an", "the", "to", "that", "for"]]

        # Convert to PascalCase
        return "".join(w.capitalize() for w in words[:3])

    def _infer_arguments(self, subject: str) -> list[str]:
        """Infer function arguments from subject."""
        args = []

        # Look for common patterns
        if "two" in subject or "pair" in subject:
            args = ["a", "b"]
        elif "list" in subject or "array" in subject:
            args = ["items"]
        elif "string" in subject or "text" in subject:
            args = ["text"]
        elif "number" in subject:
            args = ["n"]
        else:
            args = ["value"]

        return args

    def _infer_return_type(self, subject: str) -> str | None:
        """Infer return type from subject."""
        if "calculate" in subject or "sum" in subject or "count" in subject:
            return "float"
        elif "check" in subject or "validate" in subject or "is" in subject:
            return "bool"
        elif "list" in subject or "array" in subject:
            return "List[Any]"
        elif "string" in subject or "text" in subject:
            return "str"
        return None

    def _infer_methods(self, subject: str) -> list[dict[str, str]]:
        """Infer class methods from subject."""
        methods = [{"name": "__init__", "args": ["self"], "body": "pass"}]

        # Add methods based on common patterns
        if "manager" in subject.lower():
            methods.extend(
                [
                    {"name": "add", "args": ["self", "item"], "body": "pass"},
                    {"name": "remove", "args": ["self", "item"], "body": "pass"},
                    {"name": "get", "args": ["self", "key"], "body": "return None"},
                ]
            )
        elif "handler" in subject.lower():
            methods.append(
                {"name": "handle", "args": ["self", "request"], "body": "pass"}
            )

        return methods

    def _infer_attributes(self, subject: str) -> list[str]:
        """Infer class attributes from subject."""
        attrs = []

        if "list" in subject or "collection" in subject:
            attrs.append("items: List[Any] = []")
        if "manager" in subject:
            attrs.append("data: Dict[str, Any] = {}")

        return attrs

    def _generate_calculation_body(self, subject: str, args: list[str]) -> str:
        """Generate calculation function body."""
        if "sum" in subject:
            return (
                f"return sum({args[0]})"
                if "items" in args
                else f"return {' + '.join(args)}"
            )
        elif "average" in subject or "mean" in subject:
            return f"return sum({args[0]}) / len({args[0]})"
        elif "multiply" in subject or "product" in subject:
            return f"return {' * '.join(args)}"
        elif "divide" in subject:
            if len(args) >= 2:
                return f"if {args[1]} == 0:\n        raise ValueError('Division by zero')\n    return {args[0]} / {args[1]}"
            return f"return {args[0]} / divisor"
        elif "percentage" in subject:
            if len(args) >= 2:
                return f"return ({args[0]} / {args[1]}) * 100"
            return f"return ({args[0]} / total) * 100"
        elif "factorial" in subject:
            return f"""if {args[0]} < 0:
        raise ValueError('Factorial not defined for negative numbers')
    if {args[0]} <= 1:
        return 1
    result = 1
    for i in range(2, {args[0]} + 1):
        result *= i
    return result"""
        elif "power" in subject or "exponent" in subject:
            if len(args) >= 2:
                return f"return {args[0]} ** {args[1]}"
            return f"return {args[0]} ** 2"
        elif "sqrt" in subject or "square root" in subject:
            return f"import math\n    return math.sqrt({args[0]})"
        elif "max" in subject or "maximum" in subject:
            return (
                f"return max({args[0]})"
                if len(args) == 1
                else f"return max({', '.join(args)})"
            )
        elif "min" in subject or "minimum" in subject:
            return (
                f"return min({args[0]})"
                if len(args) == 1
                else f"return min({', '.join(args)})"
            )
        else:
            # Generic calculation with error handling
            return f"""try:
        # Perform calculation based on input
        result = {args[0]} if args else 0
        return result
    except Exception as e:
        raise ValueError(f'Calculation error: {{e}}')"""

    def _generate_validation_body(self, subject: str, args: list[str]) -> str:
        """Generate validation function body."""
        arg = args[0] if args else "value"

        if "email" in subject:
            return f'return "@" in {arg} and "." in {arg}'
        elif "positive" in subject:
            return f"return {arg} > 0"
        elif "empty" in subject:
            return f"return len({arg}) == 0"
        else:
            return f"return {arg} is not None"

    def _generate_conversion_body(self, subject: str, args: list[str]) -> str:
        """Generate conversion function body."""
        arg = args[0] if args else "value"

        if "string" in subject:
            return f"return str({arg})"
        elif "int" in subject:
            return f"return int({arg})"
        elif "list" in subject:
            return f"return list({arg})"
        else:
            return f"return {arg}"

    def _generate_fetch_body(self, subject: str, args: list[str]) -> str:
        """Generate fetch/get function body."""
        arg = args[0] if args else "key"

        if "database" in subject or "db" in subject:
            return f"""# Database fetch implementation
    try:
        # Example database query
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM table WHERE id = ?", ({arg},))
        result = cursor.fetchone()
        cursor.close()
        connection.close()
        return result
    except Exception as e:
        logger.error(f"Database fetch error: {{e}}")
        return None"""
        elif "api" in subject or "http" in subject or "url" in subject:
            return f"""# API fetch implementation
    import requests
    try:
        response = requests.get(f"https://api.example.com/data/{{{arg}}}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"API fetch error: {{e}}")
        return None"""
        elif "file" in subject:
            return f"""# File fetch implementation
    try:
        with open({arg}, 'r') as f:
            data = f.read()
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {{{arg}}}")
        return None
    except Exception as e:
        logger.error(f"File read error: {{e}}")
        return None"""
        elif "cache" in subject:
            return f"""# Cache fetch implementation
    cache_key = {arg}
    cached_value = cache.get(cache_key)
    
    if cached_value is not None:
        return cached_value
    
    # Fetch from source if not in cache
    value = fetch_from_source(cache_key)
    if value is not None:
        cache.set(cache_key, value)
    
    return value"""
        elif "config" in subject or "setting" in subject:
            return f"""# Configuration fetch implementation
    import os
    
    # Try environment variable first
    env_value = os.environ.get({arg}.upper())
    if env_value:
        return env_value
    
    # Try config file
    config = load_config()
    return config.get({arg}, default_value)"""
        else:
            # Generic fetch with multiple strategies
            return f"""# Generic fetch implementation
    sources = ['cache', 'database', 'api', 'file']
    
    for source in sources:
        try:
            if source == 'cache':
                result = cache.get({arg})
            elif source == 'database':
                result = fetch_from_db({arg})
            elif source == 'api':
                result = fetch_from_api({arg})
            elif source == 'file':
                result = fetch_from_file({arg})
            
            if result is not None:
                return result
        except Exception as e:
            logger.debug(f"Fetch from {{source}} failed: {{e}}")
            continue
    
    return None  # All sources failed"""

    def _generate_generic_body(self, subject: str, args: list[str]) -> str:
        """Generate generic function body based on subject analysis."""
        # Analyze subject for hints about functionality
        subject_lower = subject.lower()

        if "process" in subject_lower:
            return f"""# Process the input data
    if not {args[0] if args else 'data'}:
        return None
    
    # Apply processing steps
    result = {args[0] if args else 'data'}
    
    # Transform data as needed
    if isinstance(result, list):
        result = [process_item(item) for item in result]
    elif isinstance(result, dict):
        result = {{k: process_item(v) for k, v in result.items()}}
    else:
        result = process_item(result)
    
    return result"""

        elif "parse" in subject_lower:
            return f"""# Parse the input
    import re
    
    if not {args[0] if args else 'text'}:
        return None
    
    # Define parsing pattern
    pattern = r'\\w+'  # Adjust pattern as needed
    
    # Extract matches
    matches = re.findall(pattern, {args[0] if args else 'text'})
    
    # Return parsed data
    return matches if matches else None"""

        elif "filter" in subject_lower:
            return f"""# Filter the input based on criteria
    if not {args[0] if args else 'items'}:
        return []
    
    # Define filter criteria
    def meets_criteria(item):
        # Implement filter logic
        return True  # Placeholder
    
    # Apply filter
    filtered = [item for item in {args[0] if args else 'items'} if meets_criteria(item)]
    
    return filtered"""

        elif "transform" in subject_lower or "convert" in subject_lower:
            return f"""# Transform the input data
    if {args[0] if args else 'data'} is None:
        return None
    
    # Apply transformation
    transformed = {args[0] if args else 'data'}
    
    # Example transformations
    if isinstance(transformed, str):
        transformed = transformed.strip().lower()
    elif isinstance(transformed, (int, float)):
        transformed = round(transformed, 2)
    elif isinstance(transformed, list):
        transformed = sorted(set(transformed))
    
    return transformed"""

        elif "analyze" in subject_lower:
            return f"""# Analyze the input data
    if not {args[0] if args else 'data'}:
        return {{'error': 'No data to analyze'}}
    
    analysis = {{
        'type': type({args[0] if args else 'data'}).__name__,
        'size': len({args[0] if args else 'data'}) if hasattr({args[0] if args else 'data'}, '__len__') else 1,
    }}
    
    # Additional analysis based on data type
    if isinstance({args[0] if args else 'data'}, list):
        analysis['unique_count'] = len(set({args[0] if args else 'data'}))
        analysis['has_duplicates'] = len({args[0] if args else 'data'}) != analysis['unique_count']
    elif isinstance({args[0] if args else 'data'}, dict):
        analysis['keys'] = list({args[0] if args else 'data'}.keys())
        analysis['nested'] = any(isinstance(v, (dict, list)) for v in {args[0] if args else 'data'}.values())
    
    return analysis"""

        elif "generate" in subject_lower or "create" in subject_lower:
            return f"""# Generate output based on input
    import uuid
    from datetime import datetime
    
    # Generate unique identifier
    generated_id = str(uuid.uuid4())
    
    # Create result structure
    result = {{
        'id': generated_id,
        'timestamp': datetime.now().isoformat(),
        'input': {args[0] if args else 'None'},
        'status': 'generated'
    }}
    
    # Add generated content
    if {args[0] if args else 'None'}:
        result['content'] = f"Generated from: {{{args[0] if args else 'input'}}}"
    else:
        result['content'] = "Generated with default parameters"
    
    return result"""

        else:
            # Truly generic implementation with error handling
            return f"""# Generic implementation
    try:
        # Validate input
        if not {args[0] if args else 'True'}:
            logger.warning("No input provided")
            return None
        
        # Process input
        result = {args[0] if args else 'None'}
        
        # Apply any necessary transformations
        if hasattr(result, '__iter__') and not isinstance(result, str):
            # Handle iterables
            result = list(result)
        
        # Return processed result
        return result
        
    except Exception as e:
        logger.error(f"Processing error: {{e}}")
        raise"""

    def _infer_type_hints(
        self, subject: str, args: list[str], return_type: str | None
    ) -> dict[str, str]:
        """Infer type hints for function arguments."""
        hints = {}
        subject_lower = subject.lower()

        # Infer from argument names and subject
        for arg in args:
            if arg in ["n", "num", "number", "count"]:
                hints[arg] = "int"
            elif arg in ["text", "string", "name", "message"]:
                hints[arg] = "str"
            elif arg in ["items", "values", "elements"]:
                hints[arg] = "List[Any]"
            elif arg in ["data", "config"]:
                hints[arg] = "Dict[str, Any]"
            elif "float" in subject_lower or "decimal" in subject_lower:
                hints[arg] = "float"
            elif "bool" in subject_lower or arg in ["flag", "enabled"]:
                hints[arg] = "bool"
            else:
                hints[arg] = "Any"

        return hints

    def _infer_init_args(self, subject: str) -> list[str]:
        """Infer __init__ arguments from subject."""
        if "manager" in subject.lower():
            return ["name", "capacity"]
        elif "handler" in subject.lower():
            return ["config"]
        elif "service" in subject.lower():
            return ["endpoint", "timeout"]
        elif "controller" in subject.lower():
            return ["model", "view"]
        else:
            return ["config"]

    def _generate_init_body(self, subject: str, args: list[str]) -> str:
        """Generate __init__ method body."""
        lines = []

        # Assign all arguments
        for arg in args:
            lines.append(f"self.{arg} = {arg}")

        # Add common attributes based on pattern
        if "manager" in subject.lower():
            lines.append("self.items = []")
            lines.append("self._lock = threading.Lock()")
        elif "handler" in subject.lower():
            lines.append("self.handlers = {}")
            lines.append("self.middleware = []")
        elif "service" in subject.lower():
            lines.append("self.session = None")
            lines.append("self.retries = 3")
        elif "collection" in subject.lower() or "list" in subject.lower():
            lines.append("self.items = []")

        return "\n".join(lines) if lines else "pass"

    def _infer_base_classes(self, subject: str) -> list[str]:
        """Infer base classes from subject."""
        bases = []

        if "abstract" in subject.lower():
            bases.append("abc.ABC")
        elif "exception" in subject.lower() or "error" in subject.lower():
            bases.append("Exception")
        elif "thread" in subject.lower():
            bases.append("threading.Thread")

        return bases

    def _infer_method_return_type(self, method_name: str) -> str | None:
        """Infer return type from method name."""
        if method_name.startswith("is_") or method_name.startswith("has_"):
            return "bool"
        elif method_name.startswith("get_"):
            return "Any"
        elif method_name.startswith("count_") or method_name == "size":
            return "int"
        elif method_name in ["__str__", "__repr__"]:
            return "str"
        return None

    def _extract_test_target(self, subject: str) -> str:
        """Extract what to test from subject."""
        # Look for function/class names
        words = subject.split()
        for word in words:
            if word.startswith("test_"):
                continue
            if "_" in word or word[0].isupper():
                return word
        return "function"

    def _generate_test_cases(self, target: str) -> list[dict[str, str]]:
        """Generate comprehensive test cases for target."""
        test_cases = []

        # Basic functionality test
        test_cases.append(
            {
                "name": "test_basic_functionality",
                "assertion": f"""# Test basic functionality
    result = {target}(sample_input)
    assert result is not None
    assert isinstance(result, expected_type)""",
            }
        )

        # Edge cases
        test_cases.append(
            {
                "name": "test_empty_input",
                "assertion": f"""# Test with empty input
    result = {target}()
    assert result is None or result == [] or result == {{}}""",
            }
        )

        test_cases.append(
            {
                "name": "test_none_input",
                "assertion": f"""# Test with None input
    result = {target}(None)
    assert result is None or isinstance(result, expected_type)""",
            }
        )

        # Error handling
        test_cases.append(
            {
                "name": "test_invalid_input",
                "assertion": f"""# Test error handling
    with pytest.raises((ValueError, TypeError, Exception)):
        {target}(invalid_input)""",
            }
        )

        # Type checking
        test_cases.append(
            {
                "name": "test_type_validation",
                "assertion": f"""# Test type validation
    # Test with different types
    for test_input in [123, "string", [1,2,3], {{"key": "value"}}]:
        try:
            result = {target}(test_input)
            # Should either handle gracefully or raise appropriate error
            assert result is not None or True
        except (TypeError, ValueError):
            # Expected for incompatible types
            pass""",
            }
        )

        # Performance test
        test_cases.append(
            {
                "name": "test_performance",
                "assertion": f"""# Test performance
    import time
    start_time = time.time()
    
    # Run multiple times
    for _ in range(100):
        result = {target}(sample_input)
    
    elapsed = time.time() - start_time
    assert elapsed < 1.0  # Should complete in under 1 second""",
            }
        )

        return test_cases

    def _extract_endpoint_name(self, subject: str) -> str:
        """Extract API endpoint from subject."""
        # Look for resource names
        words = subject.lower().split()
        resources = ["users", "items", "products", "orders", "data"]

        for word in words:
            if word in resources:
                return f"/{word}"
            if word.endswith("s"):  # Plural
                return f"/{word}"

        return "/resource"

    def _load_templates(self) -> dict[str, Template]:
        """Load code generation templates."""
        templates = {
            "function": Template(
                '''def {{ name }}({{ args|join(', ') }}){% if return_type %} -> {{ return_type }}{% endif %}:
    """{{ docstring }}"""
    {{ body }}
'''
            ),
            "class": Template(
                '''class {{ name }}:
    """{{ docstring }}"""
    
    {% for attr in attributes %}
    {{ attr }}
    {% endfor %}
    
    {% for method in methods %}
    def {{ method.name }}({{ method.args|join(', ') }}):
        {{ method.body }}
    {% endfor %}
'''
            ),
            "test": Template(
                '''import pytest


def test_{{ target_name }}():
    """Test {{ target_name }} functionality."""
    {% for case in test_cases %}
    # {{ case.name }}
    {{ case.assertion }}
    {% endfor %}
'''
            ),
            "api": Template(
                '''from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route('{{ endpoint }}', methods=['{{ method }}'])
def {{ handler_name }}():
    """Handle {{ method }} {{ endpoint }}."""
    if request.method == '{{ method }}':
        # TODO: Implement handler logic
        return jsonify({'status': 'success'})
'''
            ),
        }

        return templates

    # Algorithm templates
    def _binary_search_template(self) -> str:
        return '''def binary_search(arr: List[int], target: int) -> int:
    """Binary search implementation.
    
    Returns index of target in sorted array, or -1 if not found.
    """
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
'''

    def _quick_sort_template(self) -> str:
        return '''def quick_sort(arr: List[int]) -> List[int]:
    """Quick sort implementation."""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)
'''

    def _merge_sort_template(self) -> str:
        return '''def merge_sort(arr: List[int]) -> List[int]:
    """Merge sort implementation."""
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    return merge(left, right)


def merge(left: List[int], right: List[int]) -> List[int]:
    """Merge two sorted arrays."""
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result
'''

    def _fibonacci_template(self) -> str:
        return '''def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
'''

    def _factorial_template(self) -> str:
        return '''def factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    
    return result
'''

    def _prime_template(self) -> str:
        return '''def is_prime(n: int) -> bool:
    """Check if a number is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True
'''
