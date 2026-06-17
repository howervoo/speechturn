"""Read public Python contracts without importing optional or executable modules."""

import ast
from pathlib import Path


def _signature(node):
    return {
        "arguments": ast.unparse(node.args),
        "returns": ast.unparse(node.returns) if node.returns else None,
        "decorators": [ast.unparse(value) for value in node.decorator_list],
        "async": isinstance(node, ast.AsyncFunctionDef),
    }


def module_contract(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    contract = {"exports": None, "functions": {}, "classes": {}, "constants": {}, "imports": {}}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith(
            "_"
        ):
            contract["functions"].setdefault(node.name, []).append(_signature(node))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            fields = {}
            methods = {}
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    if not child.target.id.startswith("_"):
                        fields[child.target.id] = {
                            "type": ast.unparse(child.annotation),
                            "default": ast.unparse(child.value) if child.value else None,
                        }
                elif isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name) and not target.id.startswith("_"):
                            fields[target.id] = {"value": ast.unparse(child.value)}
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not child.name.startswith("_") or (
                        child.name.startswith("__") and child.name.endswith("__")
                    ):
                        methods.setdefault(child.name, []).append(_signature(child))
            contract["classes"][node.name] = {
                "bases": [ast.unparse(base) for base in node.bases],
                "decorators": [ast.unparse(value) for value in node.decorator_list],
                "fields": fields,
                "methods": methods,
            }
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "__all__":
                    contract["exports"] = ast.literal_eval(node.value)
                elif not target.id.startswith("_") or target.id == "__version__":
                    contract["constants"][target.id] = (
                        ast.unparse(node.value) if node.value else None
                    )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                local = alias.asname or alias.name
                contract["imports"][local] = (
                    "." * node.level + str(node.module or "") + ":" + alias.name
                )
    exported = contract["exports"] or []
    contract["imports"] = {
        name: value for name, value in contract["imports"].items() if name in exported
    }
    return contract
