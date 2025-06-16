"""AST-based code generation for production-quality Python code."""
import ast
import logging
import textwrap
from dataclasses import dataclass
from typing import Any

import black

from .code_understanding import CodeAnalyzer, CodeValidator

logger = logging.getLogger(__name__)
None


@dataclass
class FunctionSpec:
    """Specification for a function to generate."""

    name: str
    args: list[str]
    return_type: str | None
    docstring: str
    body_nodes: list[ast.AST]
    decorators: list[str] = None
    is_async: bool = False
    type_hints: dict[str, str] = None


@dataclass
class ClassSpec:
    """Specification for a class to generate."""

    name: str
    bases: list[str]
    docstring: str
    attributes: dict[str, Any]
    methods: list[FunctionSpec]
    decorators: list[str] = None
    metaclass: str | None = None


class ASTCodeGenerator:
    """Generate Python code using AST manipulation for correctness."""

    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.validator = CodeValidator()

    def generate_function(self, spec: FunctionSpec) -> str:
        """Generate a function from specification using AST."""
        args = []
        for _i, arg_name in enumerate(spec.args):
            arg = ast.arg(arg=arg_name, annotation=None)
            if spec.type_hints and arg_name in spec.type_hints:
                arg.annotation = self._parse_type_annotation(spec.type_hints[arg_name])
            args.append(arg)
        arguments = ast.arguments(
            posonlyargs=[],
            args=args,
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        )
        body = []
        if spec.docstring:
            docstring_node = ast.Expr(value=ast.Constant(value=spec.docstring))
            body.append(docstring_node)
        body.extend(spec.body_nodes)
        if len(body) == 0 or len(body) == 1 and isinstance(body[0], ast.Expr):
            body.append(ast.Pass())
        returns = None
        if spec.return_type:
            returns = self._parse_type_annotation(spec.return_type)
        if spec.is_async:
            func_node = ast.AsyncFunctionDef(
                name=spec.name,
                args=arguments,
                body=body,
                decorator_list=[],
                returns=returns,
                type_comment=None,
            )
        else:
            func_node = ast.FunctionDef(
                name=spec.name,
                args=arguments,
                body=body,
                decorator_list=[],
                returns=returns,
                type_comment=None,
            )
        if spec.decorators:
            for dec in spec.decorators:
                dec_node = self._parse_decorator(dec)
                func_node.decorator_list.append(dec_node)
        module = ast.Module(body=[func_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = ast.unparse(module)
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.debug(f"Ignored exception in {'ast_code_generator.py'}: {e}")
        return code

    def generate_class(self, spec: ClassSpec) -> str:
        """Generate a class from specification using AST."""
        body = []
        if spec.docstring:
            docstring_node = ast.Expr(value=ast.Constant(value=spec.docstring))
            body.append(docstring_node)
        for attr_name, attr_value in spec.attributes.items():
            if isinstance(attr_value, str) and ":" in attr_value:
                target = ast.AnnAssign(
                    target=ast.Name(id=attr_name, ctx=ast.Store()),
                    annotation=self._parse_type_annotation(
                        attr_value.split(":")[1].strip()
                    ),
                    value=None,
                    simple=1,
                )
                body.append(target)
            else:
                assign = ast.Assign(
                    targets=[ast.Name(id=attr_name, ctx=ast.Store())],
                    value=self._value_to_ast(attr_value),
                    type_comment=None,
                )
                body.append(assign)
        for method_spec in spec.methods:
            method_code = self.generate_function(method_spec)
            method_ast = ast.parse(method_code).body[0]
            body.append(method_ast)
        if not body:
            body.append(ast.Pass())
        bases = []
        for base in spec.bases:
            if "." in base:
                parts = base.split(".")
                node = ast.Name(id=parts[0], ctx=ast.Load())
                for part in parts[1:]:
                    node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
                bases.append(node)
            else:
                bases.append(ast.Name(id=base, ctx=ast.Load()))
        keywords = []
        if spec.metaclass:
            keywords.append(
                ast.keyword(
                    arg="metaclass", value=ast.Name(id=spec.metaclass, ctx=ast.Load())
                )
            )
        class_node = ast.ClassDef(
            name=spec.name, bases=bases, keywords=keywords, body=body, decorator_list=[]
        )
        if spec.decorators:
            for dec in spec.decorators:
                dec_node = self._parse_decorator(dec)
                class_node.decorator_list.append(dec_node)
        module = ast.Module(body=[class_node], type_ignores=[])
        ast.fix_missing_locations(module)
        code = ast.unparse(module)
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.debug(f"Ignored exception in {'ast_code_generator.py'}: {e}")
        return code

    def generate_module(
        self, imports: list[str], body_elements: list[FunctionSpec | ClassSpec | str]
    ) -> str:
        """Generate a complete module with imports and multiple elements."""
        module_body = []
        for imp in imports:
            import_node = self._parse_import(imp)
            if import_node:
                module_body.append(import_node)
        if module_body and body_elements:
            pass
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
                try:
                    nodes = ast.parse(element).body
                    module_body.extend(nodes)
                except Exception as e:
                    logger.debug(f"Ignored exception in {'ast_code_generator.py'}: {e}")
        module = ast.Module(body=module_body, type_ignores=[])
        ast.fix_missing_locations(module)
        code = ast.unparse(module)
        try:
            code = black.format_str(code, mode=black.Mode())
        except Exception as e:
            logger.debug(f"Ignored exception in {'ast_code_generator.py'}: {e}")
        return code

    def _parse_type_annotation(self, type_str: str) -> ast.AST:
        """Parse a type annotation string to AST."""
        try:
            node = ast.parse(type_str, mode="eval").body
            return node
        except Exception:
            return ast.Name(id=type_str, ctx=ast.Load())

    def _parse_decorator(self, dec_str: str) -> ast.AST:
        """Parse a decorator string to AST."""
        if "(" in dec_str:
            try:
                node = ast.parse(f"@{dec_str}\ndef f(): pass").body[0].decorator_list[0]
                return node
            except Exception as e:
                logger.debug(f"Ignored exception in {'ast_code_generator.py'}: {e}")
        if "." in dec_str:
            parts = dec_str.split(".")
            node = ast.Name(id=parts[0], ctx=ast.Load())
            for part in parts[1:]:
                node = ast.Attribute(value=node, attr=part, ctx=ast.Load())
            return node
        else:
            return ast.Name(id=dec_str, ctx=ast.Load())

    def _parse_import(self, imp_str: str) -> ast.AST | None:
        """Parse an import string to AST."""
        try:
            node = ast.parse(imp_str).body[0]
            return node
        except Exception:
            if imp_str.startswith("from "):
                parts = imp_str[5:].split(" import ")
                if len(parts) == 2:
                    module = parts[0].strip()
                    names = [n.strip() for n in parts[1].split(",")]
                    return ast.ImportFrom(
                        module=module,
                        names=[ast.alias(name=n, asname=None) for n in names],
                        level=0,
                    )
            elif imp_str.startswith("import "):
                names = [n.strip() for n in imp_str[7:].split(",")]
                return ast.Import(names=[ast.alias(name=n, asname=None) for n in names])
        return None

    def _value_to_ast(self, value: Any) -> ast.AST:
        """Convert a Python value to AST node."""
        if value is None:
            return ast.Constant(value=None)
        elif isinstance(value, int | float | str | bool):
            return ast.Constant(value=value)
        elif isinstance(value, list):
            return ast.List(elts=[self._value_to_ast(v) for v in value], ctx=ast.Load())
        elif isinstance(value, dict):
            return ast.Dict(
                keys=[ast.Constant(value=k) for k in value],
                values=[self._value_to_ast(v) for v in value.values()],
            )
        elif isinstance(value, tuple):
            return ast.Tuple(
                elts=[self._value_to_ast(v) for v in value], ctx=ast.Load()
            )
        else:
            return ast.parse(repr(value), mode="eval").body

    def create_function_body(self, body_code: str) -> list[ast.AST]:
        """Parse body code into AST nodes."""
        body_code = textwrap.dedent(body_code)
        try:
            tree = ast.parse(body_code)
            return tree.body
        except SyntaxError:
            return [ast.Pass()]


def example():
    generator = ASTCodeGenerator()
    func_spec = FunctionSpec(
        name="calculate_fibonacci",
        args=["n"],
        return_type="int",
        docstring="Calculate the nth Fibonacci number.",
        body_nodes=generator.create_function_body(
            """
            if n <= 1:
                return n
            a, b = 0, 1
            for _ in range(2, n + 1):
                a, b = b, a + b
            return b
        """
        ),
        type_hints={"n": "int"},
    )
    func_code = generator.generate_function(func_spec)
    print("Generated function:")
    print(func_code)
    init_spec = FunctionSpec(
        name="__init__",
        args=['self", "name'],
        return_type=None,
        docstring="Initialize the manager.",
        body_nodes=generator.create_function_body(
            """self.name = name
self.items = []"""
        ),
        type_hints={"name": "str"},
    )
    add_spec = FunctionSpec(
        name="add_item",
        args=['self", "item'],
        return_type=None,
        docstring="Add an item.",
        body_nodes=generator.create_function_body("self.items.append(item)"),
        type_hints={"item": "Any"},
    )
    class_spec = ClassSpec(
        name="ItemManager",
        bases=[],
        docstring="Manages a collection of items.",
        attributes={"items": "List[Any]"},
        methods=[init_spec, add_spec],
    )
    class_code = generator.generate_class(class_spec)
    print("\nGenerated class:")
    print(class_code)


if __name__ == "__main__":
    example()
