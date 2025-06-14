"""AST-based code generation for production-quality Python code."""
import ast
from typing import List, Dict, Optional, Any, Union
import black
from dataclasses import dataclass
import textwrap

from .code_understanding import CodeAnalyzer, CodeValidator


@dataclass
class FunctionSpec:
    """Specification for a function to generate."""
    name: str
    args: List[str]
    return_type: Optional[str]
    docstring: str
    body_nodes: List[ast.AST]
    decorators: List[str] = None
    is_async: bool = False
    type_hints: Dict[str, str] = None


@dataclass 
class ClassSpec:
    """Specification for a class to generate."""
    name: str
    bases: List[str]
    docstring: str
    attributes: Dict[str, Any]
    methods: List[FunctionSpec]
    decorators: List[str] = None
    metaclass: Optional[str] = None


class ASTCodeGenerator:
    """Generate Python code using AST manipulation for correctness."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.validator = CodeValidator()
    
    def generate_function(self, spec: FunctionSpec) -> str:
        """Generate a function from specification using AST."""
        # Build arguments
        args = []
        for i, arg_name in enumerate(spec.args):
            arg = ast.arg(arg=arg_name, annotation=None)
            
            # Add type hint if available
            if spec.type_hints and arg_name in spec.type_hints:
                arg.annotation = self._parse_type_annotation(spec.type_hints[arg_name])
            
            args.append(arg)
        
        # Build function arguments
        arguments = ast.arguments(
            posonlyargs=[],
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[]
        )
        
        # Build function body
        body = []
        
        # Add docstring
        if spec.docstring:
            docstring_node = ast.Expr(value=ast.Constant(value=spec.docstring))
            body.append(docstring_node)
        
        # Add body nodes
        body.extend(spec.body_nodes)
        
        # Ensure we have at least one statement
        if len(body) == 0 or (len(body) == 1 and isinstance(body[0], ast.Expr)):
            body.append(ast.Pass())
        
        # Build return annotation
        returns = None
        if spec.return_type:
            returns = self._parse_type_annotation(spec.return_type)
        
        # Create function def
        if spec.is_async:
            func_node = ast.AsyncFunctionDef(
                name=spec.name,
                args=arguments,
                body=body,
                decorator_list=[],
                returns=returns,
                type_comment=None
            )
        else:
            func_node = ast.FunctionDef(
                name=spec.name,
                args=arguments,
                body=body,
                decorator_list=[],
                returns=returns,
                type_comment=None
            )
        
        # Add decorators
        if spec.decorators:
            for dec in spec.decorators:
                dec_node = self._parse_decorator(dec)
                func_node.decorator_list.append(dec_node)
        
        # Convert to code
        module = ast.Module(body=[func_node], type_ignores=[])
        ast.fix_missing_locations(module)
        
        # Generate code
        code = ast.unparse(module)
        
        # Format with black
        try:
            code = black.format_str(code, mode=black.Mode())
        except:
            pass
        
        return code
    
    def generate_class(self, spec: ClassSpec) -> str:
        """Generate a class from specification using AST."""
        # Build class body
        body = []
        
        # Add docstring
        if spec.docstring:
            docstring_node = ast.Expr(value=ast.Constant(value=spec.docstring))
            body.append(docstring_node)
        
        # Add class attributes
        for attr_name, attr_value in spec.attributes.items():
            # Create assignment
            if isinstance(attr_value, str) and ':' in attr_value:
                # Type annotation
                target = ast.AnnAssign(
                    target=ast.Name(id=attr_name, ctx=ast.Store()),
                    annotation=self._parse_type_annotation(attr_value.split(':')[1].strip()),
                    value=None,
                    simple=1
                )
                body.append(target)
            else:
                # Regular assignment
                assign = ast.Assign(
                    targets=[ast.Name(id=attr_name, ctx=ast.Store())],
                    value=self._value_to_ast(attr_value),
                    type_comment=None
                )
                body.append(assign)
        
        # Add methods
        for method_spec in spec.methods:
            method_code = self.generate_function(method_spec)
            method_ast = ast.parse(method_code).body[0]
            body.append(method_ast)
        
        # Ensure we have at least one statement
        if not body:
            body.append(ast.Pass())
        
        # Build bases
        bases = []
        for base in spec.bases:
            if '.' in base:
                # Attribute access (e.g., abc.ABC)
                parts = base.split('.')
                node = ast.Name(id=parts[0], ctx=ast.Load())
                for part in parts[1:]:
                    node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
                bases.append(node)
            else:
                bases.append(ast.Name(id=base, ctx=ast.Load()))
        
        # Build keywords (for metaclass)
        keywords = []
        if spec.metaclass:
            keywords.append(ast.keyword(
                arg='metaclass',
                value=ast.Name(id=spec.metaclass, ctx=ast.Load())
            ))
        
        # Create class def
        class_node = ast.ClassDef(
            name=spec.name,
            bases=bases,
            keywords=keywords,
            body=body,
            decorator_list=[]
        )
        
        # Add decorators
        if spec.decorators:
            for dec in spec.decorators:
                dec_node = self._parse_decorator(dec)
                class_node.decorator_list.append(dec_node)
        
        # Convert to code
        module = ast.Module(body=[class_node], type_ignores=[])
        ast.fix_missing_locations(module)
        
        # Generate code
        code = ast.unparse(module)
        
        # Format with black
        try:
            code = black.format_str(code, mode=black.Mode())
        except:
            pass
        
        return code
    
    def generate_module(self, imports: List[str], body_elements: List[Union[FunctionSpec, ClassSpec, str]]) -> str:
        """Generate a complete module with imports and multiple elements."""
        module_body = []
        
        # Add imports
        for imp in imports:
            import_node = self._parse_import(imp)
            if import_node:
                module_body.append(import_node)
        
        # Add blank line after imports if any
        if module_body and body_elements:
            # This will be handled by black formatter
            pass
        
        # Add body elements
        for element in body_elements:
            if isinstance(element, FunctionSpec):
                code = self.generate_function(element)
                node = ast.parse(code).body[0]
                module_body.append(node)
            elif isinstance(element, ClassSpec):
                code = self.generate_class(element)
                node = ast.parse(code).body[0]
                module_body.append(node)
            elif isinstance(element, str):
                # Parse as statement
                try:
                    nodes = ast.parse(element).body
                    module_body.extend(nodes)
                except:
                    # Add as comment
                    pass
        
        # Create module
        module = ast.Module(body=module_body, type_ignores=[])
        ast.fix_missing_locations(module)
        
        # Generate code
        code = ast.unparse(module)
        
        # Format with black
        try:
            code = black.format_str(code, mode=black.Mode())
        except:
            pass
        
        return code
    
    def _parse_type_annotation(self, type_str: str) -> ast.AST:
        """Parse a type annotation string to AST."""
        try:
            # Parse as expression
            node = ast.parse(type_str, mode='eval').body
            return node
        except:
            # Fallback to Name node
            return ast.Name(id=type_str, ctx=ast.Load())
    
    def _parse_decorator(self, dec_str: str) -> ast.AST:
        """Parse a decorator string to AST."""
        if '(' in dec_str:
            # Decorator with arguments
            try:
                node = ast.parse(f"@{dec_str}\ndef f(): pass").body[0].decorator_list[0]
                return node
            except:
                pass
        
        # Simple decorator
        if '.' in dec_str:
            parts = dec_str.split('.')
            node = ast.Name(id=parts[0], ctx=ast.Load())
            for part in parts[1:]:
                node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
            return node
        else:
            return ast.Name(id=dec_str, ctx=ast.Load())
    
    def _parse_import(self, imp_str: str) -> Optional[ast.AST]:
        """Parse an import string to AST."""
        try:
            # Try parsing directly
            node = ast.parse(imp_str).body[0]
            return node
        except:
            # Build import manually
            if imp_str.startswith('from '):
                # from X import Y
                parts = imp_str[5:].split(' import ')
                if len(parts) == 2:
                    module = parts[0].strip()
                    names = [n.strip() for n in parts[1].split(',')]
                    
                    return ast.ImportFrom(
                        module=module,
                        names=[ast.alias(name=n, asname=None) for n in names],
                        level=0
                    )
            elif imp_str.startswith('import '):
                # import X
                names = [n.strip() for n in imp_str[7:].split(',')]
                return ast.Import(
                    names=[ast.alias(name=n, asname=None) for n in names]
                )
        
        return None
    
    def _value_to_ast(self, value: Any) -> ast.AST:
        """Convert a Python value to AST node."""
        if value is None:
            return ast.Constant(value=None)
        elif isinstance(value, (int, float, str, bool)):
            return ast.Constant(value=value)
        elif isinstance(value, list):
            return ast.List(
                elts=[self._value_to_ast(v) for v in value],
                ctx=ast.Load()
            )
        elif isinstance(value, dict):
            return ast.Dict(
                keys=[ast.Constant(value=k) for k in value.keys()],
                values=[self._value_to_ast(v) for v in value.values()]
            )
        elif isinstance(value, tuple):
            return ast.Tuple(
                elts=[self._value_to_ast(v) for v in value],
                ctx=ast.Load()
            )
        else:
            # Fallback to string representation
            return ast.parse(repr(value), mode='eval').body
    
    def create_function_body(self, body_code: str) -> List[ast.AST]:
        """Parse body code into AST nodes."""
        # Dedent if needed
        body_code = textwrap.dedent(body_code)
        
        try:
            # Parse the body
            tree = ast.parse(body_code)
            return tree.body
        except SyntaxError:
            # Return simple pass statement
            return [ast.Pass()]


# Example usage
def example():
    generator = ASTCodeGenerator()
    
    # Generate a function
    func_spec = FunctionSpec(
        name="calculate_fibonacci",
        args=["n"],
        return_type="int",
        docstring="Calculate the nth Fibonacci number.",
        body_nodes=generator.create_function_body("""
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        """),
        type_hints={"n": "int"}
    )
    
    func_code = generator.generate_function(func_spec)
    print("Generated function:")
    print(func_code)
    
    # Generate a class
    init_spec = FunctionSpec(
        name="__init__",
        args=["self", "name"],
        return_type=None,
        docstring="Initialize the manager.",
        body_nodes=generator.create_function_body("self.name = name\nself.items = []"),
        type_hints={"name": "str"}
    )
    
    add_spec = FunctionSpec(
        name="add_item",
        args=["self", "item"],
        return_type=None,
        docstring="Add an item.",
        body_nodes=generator.create_function_body("self.items.append(item)"),
        type_hints={"item": "Any"}
    )
    
    class_spec = ClassSpec(
        name="ItemManager",
        bases=[],
        docstring="Manages a collection of items.",
        attributes={"items": "List[Any]"},
        methods=[init_spec, add_spec]
    )
    
    class_code = generator.generate_class(class_spec)
    print("\nGenerated class:")
    print(class_code)


if __name__ == "__main__":
    example()