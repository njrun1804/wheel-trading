"""Real code understanding using AST and libcst."""
import ast
import libcst as cst
from typing import List, Dict, Any, Optional, Tuple
import black
from dataclasses import dataclass
import textwrap


@dataclass
class CodeContext:
    """Context extracted from code."""
    imports: List[str]
    functions: List[Dict[str, Any]]
    classes: List[Dict[str, Any]]
    variables: List[str]
    docstring: Optional[str]
    complexity: int
    type_hints: Dict[str, str]


class CodeAnalyzer:
    """Analyzes Python code to understand structure and context."""
    
    def analyze(self, code: str) -> CodeContext:
        """Extract comprehensive context from code."""
        try:
            tree = ast.parse(code)
            module = cst.parse_module(code)
            
            imports = self._extract_imports(tree)
            functions = self._extract_functions(tree)
            classes = self._extract_classes(tree)
            variables = self._extract_variables(tree)
            docstring = ast.get_docstring(tree)
            complexity = self._calculate_complexity(tree)
            type_hints = self._extract_type_hints(module)
            
            return CodeContext(
                imports=imports,
                functions=functions,
                classes=classes,
                variables=variables,
                docstring=docstring,
                complexity=complexity,
                type_hints=type_hints
            )
        except SyntaxError:
            # Return empty context for invalid code
            return CodeContext([], [], [], [], None, 0, {})
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all imports from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        return imports
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions with metadata."""
        functions = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = {
                    'name': node.name,
                    'args': [arg.arg for arg in node.args.args],
                    'defaults': len(node.args.defaults),
                    'docstring': ast.get_docstring(node),
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                    'returns': self._get_annotation(node.returns),
                    'is_async': isinstance(node, ast.AsyncFunctionDef),
                    'lineno': node.lineno
                }
                functions.append(func_info)
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions with metadata."""
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item.name)
                
                class_info = {
                    'name': node.name,
                    'bases': [self._get_name(base) for base in node.bases],
                    'methods': methods,
                    'docstring': ast.get_docstring(node),
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                    'lineno': node.lineno
                }
                classes.append(class_info)
        return classes
    
    def _extract_variables(self, tree: ast.AST) -> List[str]:
        """Extract variable assignments."""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        variables.append(target.id)
        return list(set(variables))  # Unique variables
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity
    
    def _extract_type_hints(self, module: cst.Module) -> Dict[str, str]:
        """Extract type hints using libcst."""
        type_hints = {}
        
        class TypeHintCollector(cst.CSTVisitor):
            def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
                if node.returns:
                    type_hints[f"{node.name}_return"] = node.returns.annotation.value
                
                for param in node.params.params:
                    if param.annotation:
                        type_hints[f"{node.name}_{param.name.value}"] = param.annotation.annotation.value
        
        module.walk(TypeHintCollector())
        return type_hints
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Get decorator name as string."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_name(decorator.value)}.{decorator.attr}"
        elif isinstance(decorator, ast.Call):
            return self._get_decorator_name(decorator.func)
        return "unknown"
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from various AST nodes."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return "unknown"
    
    def _get_annotation(self, node: Optional[ast.AST]) -> Optional[str]:
        """Get type annotation as string."""
        if not node:
            return None
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return str(node.value)
        return ast.unparse(node) if hasattr(ast, 'unparse') else None


class CodeTransformer:
    """Transform code using libcst while preserving structure."""
    
    def add_function(self, code: str, func_name: str, func_body: str, 
                    args: List[str] = None, return_type: str = None,
                    docstring: str = None) -> str:
        """Add a new function to code."""
        module = cst.parse_module(code)
        
        # Build function parameters
        params = []
        for arg in (args or []):
            param = cst.Param(name=cst.Name(arg))
            params.append(param)
        
        # Build function body
        body_stmts = []
        if docstring:
            body_stmts.append(
                cst.SimpleStatementLine(body=[
                    cst.Expr(cst.SimpleString(f'"""{docstring}"""'))
                ])
            )
        
        # Parse function body
        body_module = cst.parse_module(textwrap.dedent(func_body))
        body_stmts.extend(body_module.body)
        
        # Create function def
        func_def = cst.FunctionDef(
            name=cst.Name(func_name),
            params=cst.Parameters(params=params),
            body=cst.IndentedBlock(body=body_stmts),
            returns=cst.Annotation(annotation=cst.Name(return_type)) if return_type else None
        )
        
        # Add to module
        new_module = module.with_changes(
            body=[*module.body, cst.SimpleStatementLine(body=[]), func_def]
        )
        
        return new_module.code
    
    def add_import(self, code: str, import_name: str, from_module: str = None) -> str:
        """Add an import statement to code."""
        module = cst.parse_module(code)
        
        if from_module:
            import_stmt = cst.SimpleStatementLine(body=[
                cst.ImportFrom(
                    module=cst.Attribute(value=cst.Name(from_module.split('.')[0]),
                                       attr=cst.Name('.'.join(from_module.split('.')[1:])))
                    if '.' in from_module else cst.Name(from_module),
                    names=[cst.ImportAlias(name=cst.Name(import_name))]
                )
            ])
        else:
            import_stmt = cst.SimpleStatementLine(body=[
                cst.Import(names=[cst.ImportAlias(name=cst.Name(import_name))])
            ])
        
        # Add import at the beginning
        new_body = [import_stmt, *module.body]
        new_module = module.with_changes(body=new_body)
        
        return new_module.code
    
    def add_type_hints(self, code: str) -> str:
        """Add type hints to functions based on usage."""
        module = cst.parse_module(code)
        
        class TypeHintAdder(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node: cst.FunctionDef, 
                                updated_node: cst.FunctionDef) -> cst.FunctionDef:
                # Simple heuristic: add Any type if no type hint
                new_params = []
                for param in updated_node.params.params:
                    if not param.annotation:
                        new_param = param.with_changes(
                            annotation=cst.Annotation(annotation=cst.Name("Any"))
                        )
                        new_params.append(new_param)
                    else:
                        new_params.append(param)
                
                return updated_node.with_changes(
                    params=updated_node.params.with_changes(params=new_params)
                )
        
        # Add typing import if needed
        if "from typing import" not in code:
            code = self.add_import(code, "Any", "typing")
            module = cst.parse_module(code)
        
        modified_tree = module.visit(TypeHintAdder())
        return modified_tree.code
    
    def optimize_imports(self, code: str) -> str:
        """Organize and optimize imports."""
        try:
            # Use black for import sorting
            return black.format_str(code, mode=black.Mode())
        except:
            return code
    
    def add_error_handling(self, code: str, func_name: str) -> str:
        """Add try-except to a function."""
        module = cst.parse_module(code)
        
        class ErrorHandlerAdder(cst.CSTTransformer):
            def leave_FunctionDef(self, original_node: cst.FunctionDef,
                                updated_node: cst.FunctionDef) -> cst.FunctionDef:
                if updated_node.name.value != func_name:
                    return updated_node
                
                # Wrap body in try-except
                try_stmt = cst.Try(
                    body=updated_node.body,
                    handlers=[
                        cst.ExceptHandler(
                            type=cst.Name("Exception"),
                            name=cst.AsName(name=cst.Name("e")),
                            body=cst.IndentedBlock(body=[
                                cst.SimpleStatementLine(body=[
                                    cst.Raise(exc=cst.Name("e"))
                                ])
                            ])
                        )
                    ]
                )
                
                return updated_node.with_changes(
                    body=cst.IndentedBlock(body=[try_stmt])
                )
        
        modified_tree = module.visit(ErrorHandlerAdder())
        return modified_tree.code


class CodeValidator:
    """Validate generated code."""
    
    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)
    
    def validate_imports(self, code: str) -> Tuple[bool, List[str]]:
        """Check if all imports are valid."""
        missing = []
        analyzer = CodeAnalyzer()
        context = analyzer.analyze(code)
        
        for imp in context.imports:
            module_name = imp.split('.')[0]
            try:
                __import__(module_name)
            except ImportError:
                missing.append(imp)
        
        return len(missing) == 0, missing
    
    def check_complexity(self, code: str, max_complexity: int = 10) -> Tuple[bool, int]:
        """Check if code complexity is within limits."""
        analyzer = CodeAnalyzer()
        context = analyzer.analyze(code)
        return context.complexity <= max_complexity, context.complexity
    
    def format_code(self, code: str) -> str:
        """Format code using black."""
        try:
            return black.format_str(code, mode=black.Mode())
        except:
            return code