"""
Lightweight symbol index for fast repository navigation.

Indexes functions, classes, methods, imports, and exports using simple
regex-based parsing. Not a full compiler — optimized for speed and
deterministic results.
"""

from __future__ import annotations

import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class Symbol:
    """A single indexed symbol."""
    name: str
    kind: str  # 'function', 'class', 'method', 'import', 'variable'
    file_path: str
    line_number: int = 0
    col_offset: int = 0
    end_line: int = 0
    docstring: str = ''
    parent_class: str = ''  # for methods
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)


@dataclass
class ImportInfo:
    """An import statement."""
    module: str
    names: list[str]  # imported names (empty = import module)
    file_path: str
    line_number: int = 0
    is_from: bool = False  # True for `from X import Y`


class SymbolIndex:
    """Lightweight symbol index using regex-based parsing.

    Supports Python, JavaScript/TypeScript (basic), Rust (basic), Go (basic).
    Not a full parser — trades completeness for speed and determinism.
    """

    # Python patterns
    _PY_FUNC = re.compile(
        r'^(?P<indent>[ \t]*)(?:async\s+)?def\s+(?P<name>\w+)\s*\(',
        re.MULTILINE,
    )
    _PY_CLASS = re.compile(
        r'^class\s+(?P<name>\w+)(?:\([^)]*\))?:',
        re.MULTILINE,
    )
    _PY_METHOD = re.compile(
        r'^(?P<indent>[ \t]+)(?:async\s+)?def\s+(?P<name>\w+)\s*\(',
        re.MULTILINE,
    )
    _PY_FROM_IMPORT = re.compile(
        r'^from\s+(?P<module>[\w.]+)\s+import\s+(?P<names>[\w ,*]+)',
        re.MULTILINE,
    )
    _PY_IMPORT = re.compile(
        r'^import\s+(?P<names>[\w ,*]+)',
        re.MULTILINE,
    )
    _PY_DECORATOR = re.compile(
        r'^@\s*(?P<dec>\S+)',
        re.MULTILINE,
    )
    _PY_DOCSTRING = re.compile(
        r'^\s*"""(?P<doc>.*?)"""',
        re.MULTILINE | re.DOTALL,
    )

    # JS/TS patterns
    _JS_FUNC = re.compile(
        r'(?:export\s+)?(?:async\s+)?function\s+(?P<name>\w+)',
        re.MULTILINE,
    )
    _JS_CONST_FUNC = re.compile(
        r'(?:export\s+)?(?:const|let|var)\s+(?P<name>\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[\w]+)\s*=>',
        re.MULTILINE,
    )
    _JS_CLASS = re.compile(
        r'(?:export\s+)?class\s+(?P<name>\w+)',
        re.MULTILINE,
    )
    _JS_IMPORT = re.compile(
        r'import\s+(?:\{[^}]*\}|[\w*]+)\s+from\s+[\'"](?P<module>[^\'"]+)[\'"]',
        re.MULTILINE,
    )
    _JS_EXPORT = re.compile(
        r'export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(?P<name>\w+)',
        re.MULTILINE,
    )

    # Rust patterns
    _RS_FUNC = re.compile(
        r'(?:pub\s+)?(?:async\s+)?fn\s+(?P<name>\w+)',
        re.MULTILINE,
    )
    _RS_STRUCT = re.compile(
        r'(?:pub\s+)?struct\s+(?P<name>\w+)',
        re.MULTILINE,
    )
    _RS_IMPL = re.compile(
        r'impl(?:<[^>]+>)?\s+(?P<type>\w+)',
        re.MULTILINE,
    )
    _RS_USE = re.compile(
        r'use\s+(?P<module>[\w:]+)',
        re.MULTILINE,
    )

    # Go patterns
    _GO_FUNC = re.compile(
        r'func\s+(?:\([^)]+\)\s+)?(?P<name>\w+)\s*\(',
        re.MULTILINE,
    )
    _GO_TYPE = re.compile(
        r'type\s+(?P<name>\w+)\s+(?:struct|interface)',
        re.MULTILINE,
    )
    _GO_IMPORT = re.compile(
        r'(?:"([^"]+)"|(\S+))',
        re.MULTILINE,
    )

    def __init__(self) -> None:
        self._symbols: list[Symbol] = []
        self._imports: list[ImportInfo] = []
        self._by_file: dict[str, list[Symbol]] = defaultdict(list)
        self._by_name: dict[str, list[Symbol]] = defaultdict(list)
        self._by_kind: dict[str, list[Symbol]] = defaultdict(list)
        self._import_by_file: dict[str, list[ImportInfo]] = defaultdict(list)
        self._import_by_module: dict[str, list[ImportInfo]] = defaultdict(list)
        self._lock = threading.Lock()
        self._indexed_files: set[str] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_file(self, path: str | Path) -> int:
        """Index a single file. Returns the number of symbols found."""
        path = Path(path)
        path_str = str(path.resolve())

        if path_str in self._indexed_files:
            return 0

        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except (OSError, PermissionError):
            return 0

        ext = path.suffix.lower()
        symbols, imports = self._parse_file(content, path_str, ext)

        with self._lock:
            self._indexed_files.add(path_str)
            self._symbols.extend(symbols)
            self._imports.extend(imports)
            for sym in symbols:
                self._by_file[sym.file_path].append(sym)
                self._by_name[sym.name].append(sym)
                self._by_kind[sym.kind].append(sym)
            for imp in imports:
                self._import_by_file[imp.file_path].append(imp)
                self._import_by_module[imp.module].append(imp)

        return len(symbols)

    def index_directory(self, root: str | Path, extensions: Optional[set[str]] = None) -> int:
        """Index all matching files in a directory tree."""
        root = Path(root)
        if extensions is None:
            extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.rs', '.go'}

        total = 0
        for fp in root.rglob('*'):
            if fp.is_file() and fp.suffix.lower() in extensions:
                # Skip common non-source directories
                if any(part in {'.git', '__pycache__', 'node_modules', '.venv', 'venv', '.tox'}
                       for part in fp.parts):
                    continue
                total += self.index_file(fp)
        return total

    def find_definition(self, name: str) -> list[Symbol]:
        """Find where a symbol is defined."""
        with self._lock:
            return [s for s in self._by_name.get(name, [])
                    if s.kind in ('function', 'class', 'method')]

    def find_imports(self, name: str) -> list[ImportInfo]:
        """Find files that import a given module/name."""
        with self._lock:
            results = []
            for imp in self._imports:
                if name in imp.names or imp.module == name or imp.module.endswith(f'.{name}'):
                    results.append(imp)
            return results

    def get_symbols_in_file(self, path: str | Path) -> list[Symbol]:
        """Get all indexed symbols in a file."""
        path_str = str(Path(path).resolve())
        with self._lock:
            return list(self._by_file.get(path_str, []))

    def get_imports_in_file(self, path: str | Path) -> list[ImportInfo]:
        """Get all imports in a file."""
        path_str = str(Path(path).resolve())
        with self._lock:
            return list(self._import_by_file.get(path_str, []))

    def get_symbols_by_kind(self, kind: str) -> list[Symbol]:
        """Get all symbols of a given kind."""
        with self._lock:
            return list(self._by_kind.get(kind, []))

    def search(self, query: str) -> list[Symbol]:
        """Search symbols by name (case-insensitive substring match)."""
        query_lower = query.lower()
        with self._lock:
            return [s for s in self._symbols if query_lower in s.name.lower()]

    def clear(self) -> None:
        """Clear the entire index."""
        with self._lock:
            self._symbols.clear()
            self._imports.clear()
            self._by_file.clear()
            self._by_name.clear()
            self._by_kind.clear()
            self._import_by_file.clear()
            self._import_by_module.clear()
            self._indexed_files.clear()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                'files': len(self._indexed_files),
                'symbols': len(self._symbols),
                'imports': len(self._imports),
                'kinds': {k: len(v) for k, v in self._by_kind.items()},
            }

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse_file(
        self, content: str, file_path: str, ext: str
    ) -> tuple[list[Symbol], list[ImportInfo]]:
        """Parse a file and extract symbols and imports."""
        if ext == '.py':
            return self._parse_python(content, file_path)
        elif ext in ('.js', '.ts', '.jsx', '.tsx'):
            return self._parse_javascript(content, file_path)
        elif ext == '.rs':
            return self._parse_rust(content, file_path)
        elif ext == '.go':
            return self._parse_go(content, file_path)
        return [], []

    def _parse_python(self, content: str, file_path: str) -> tuple[list[Symbol], list[ImportInfo]]:
        lines = content.split('\n')
        symbols: list[Symbol] = []
        imports: list[ImportInfo] = []

        # Find all decorators
        decorators: dict[int, list[str]] = defaultdict(list)
        for m in self._PY_DECORATOR.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            decorators[line_no].append(m.group('dec'))

        # Find classes
        classes: dict[str, int] = {}  # name -> line
        for m in self._PY_CLASS.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            name = m.group('name')
            classes[name] = line_no
            symbols.append(Symbol(
                name=name,
                kind='class',
                file_path=file_path,
                line_number=line_no,
            ))

        # Find top-level functions
        for m in self._PY_FUNC.finditer(content):
            indent = m.group('indent')
            if len(indent) > 0:
                continue  # skip methods (handled below)
            line_no = content[:m.start()].count('\n') + 1
            name = m.group('name')
            is_async = 'async' in m.group(0)
            decs = decorators.get(line_no, [])
            symbols.append(Symbol(
                name=name,
                kind='function',
                file_path=file_path,
                line_number=line_no,
                is_async=is_async,
                decorators=decs,
            ))

        # Find methods inside classes
        current_class = ''
        for i, line in enumerate(lines, 1):
            for cls_name, cls_line in classes.items():
                if i == cls_line:
                    current_class = cls_name
                    break

        for m in self._PY_METHOD.finditer(content):
            indent = m.group('indent')
            if len(indent) <= 0:
                continue
            line_no = content[:m.start()].count('\n') + 1
            name = m.group('name')
            is_async = 'async' in m.group(0)

            # Determine parent class (approximate)
            parent = ''
            for cls_name, cls_line in classes.items():
                if cls_line < line_no:
                    parent = cls_name

            symbols.append(Symbol(
                name=name,
                kind='method',
                file_path=file_path,
                line_number=line_no,
                parent_class=parent,
                is_async=is_async,
            ))

        # Find imports
        for m in self._PY_FROM_IMPORT.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            module = m.group('module')
            names_raw = m.group('names')
            names = [n.strip().rstrip(',') for n in names_raw.split(',') if n.strip()]
            imports.append(ImportInfo(
                module=module,
                names=names,
                file_path=file_path,
                line_number=line_no,
                is_from=True,
            ))

        for m in self._PY_IMPORT.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            names_raw = m.group('names')
            names = [n.strip().rstrip(',') for n in names_raw.split(',') if n.strip()]
            # `import os` → module='os', names=['os']
            module = names[0] if names else ''
            imports.append(ImportInfo(
                module=module,
                names=names,
                file_path=file_path,
                line_number=line_no,
                is_from=False,
            ))

        return symbols, imports

    def _parse_javascript(self, content: str, file_path: str) -> tuple[list[Symbol], list[ImportInfo]]:
        symbols: list[Symbol] = []
        imports: list[ImportInfo] = []

        for m in self._JS_FUNC.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='function',
                file_path=file_path, line_number=line_no,
            ))

        for m in self._JS_CONST_FUNC.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='function',
                file_path=file_path, line_number=line_no,
            ))

        for m in self._JS_CLASS.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='class',
                file_path=file_path, line_number=line_no,
            ))

        for m in self._JS_IMPORT.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            imports.append(ImportInfo(
                module=m.group('module'), names=[],
                file_path=file_path, line_number=line_no,
            ))

        return symbols, imports

    def _parse_rust(self, content: str, file_path: str) -> tuple[list[Symbol], list[ImportInfo]]:
        symbols: list[Symbol] = []
        imports: list[ImportInfo] = []

        for m in self._RS_FUNC.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='function',
                file_path=file_path, line_number=line_no,
            ))

        for m in self._RS_STRUCT.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='class',
                file_path=file_path, line_number=line_no,
            ))

        for m in self._RS_USE.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            imports.append(ImportInfo(
                module=m.group('module'), names=[],
                file_path=file_path, line_number=line_no,
            ))

        return symbols, imports

    def _parse_go(self, content: str, file_path: str) -> tuple[list[Symbol], list[ImportInfo]]:
        symbols: list[Symbol] = []
        imports: list[ImportInfo] = []

        for m in self._GO_FUNC.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='function',
                file_path=file_path, line_number=line_no,
            ))

        for m in self._GO_TYPE.finditer(content):
            line_no = content[:m.start()].count('\n') + 1
            symbols.append(Symbol(
                name=m.group('name'), kind='class',
                file_path=file_path, line_number=line_no,
            ))

        return symbols, imports
