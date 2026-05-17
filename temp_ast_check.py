import ast, pathlib
root = pathlib.Path(r'F:/Joydeb-Data/EchoFace_Eng1.0.1')
for pyfile in root.rglob('*.py'):
    src = pyfile.read_text(encoding='utf-8', errors='ignore')
    try:
        tree = ast.parse(src)
    except Exception:
        continue
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            cond = node.test
            if isinstance(cond, ast.Name):
                print(pyfile, ast.get_source_segment(src,node).strip())
            elif isinstance(cond, ast.UnaryOp) and isinstance(cond.operand, ast.Name):
                print(pyfile, ast.get_source_segment(src,node).strip())
            elif isinstance(cond, ast.BoolOp) and all(isinstance(v, ast.Name) for v in cond.values):
                print(pyfile, ast.get_source_segment(src,node).strip())
