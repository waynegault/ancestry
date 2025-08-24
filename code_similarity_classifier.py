#!/usr/bin/env python3
"""
Function & Method Classifier + Similarity Detector

Purpose:
- Scan every .py file in this repo
- Extract every function and method with metadata
- Classify by behavior (pure vs I/O kinds), size, async, regex/json usage, etc.
- Compute normalized fingerprints and detect near-duplicates (candidates to merge)
- Emit a concise text summary and optionally write a JSON report

Testing:
- Provides run_comprehensive_tests() so run_all_tests.py will pick it up
- Tests assert that we discover functions and produce basic similarity candidates

No third-party deps. Pure stdlib.
"""
from __future__ import annotations

import ast
import difflib
import hashlib
import io
import json
import keyword
import re
import sys
import tokenize
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ----------------------------- Data Structures -----------------------------

@dataclass
class FunctionInfo:
    module_path: str
    qualname: str  # ClassName.func or func
    lineno: int
    end_lineno: int
    is_method: bool
    is_async: bool
    args_count: int
    has_varargs: bool
    has_kwargs: bool
    returns_count: int
    yield_count: int
    complexity: int  # rough cyclomatic proxy
    loc: int
    tags: List[str]
    signature: str
    fingerprint: str  # sha1 of normalized token stream
    simhash64: int
    normalized: str  # for similarity matching


@dataclass
class SimilarityPair:
    a_idx: int
    b_idx: int
    score: float
    hamming: int
    reason: str


# ----------------------------- Analyzer Core -------------------------------

IGNORED_PARTS = (
    "__pycache__",
    ".venv",
    "site-packages",
    "Logs/",
    "Logs\\",
    "Cache",
)

SKIP_FILE_PATTERNS = (
    "_backup.py",
    "_temp.py",
    "prototype",
    "experimental",
    "sandbox",
)

IO_KEYWORDS = {
    "open", "print", "input", "requests", "urllib", "httpx", "selenium",
    "subprocess", "sqlalchemy", "sqlite3", "psycopg2", "Path", "os", "shutil",
}


def _safe_read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _end_lineno(node: ast.AST) -> int:
    return getattr(node, "end_lineno", getattr(node, "lineno", 0))


def _rough_complexity(node: ast.AST) -> int:
    # simple proxy: 1 + decision points
    count = 1
    for sub in ast.walk(node):
        if isinstance(sub, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.BoolOp)) or isinstance(sub, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            count += 1
    return count


def _collect_tags(src: str, node: ast.AST, is_async: bool) -> List[str]:
    tags: List[str] = []
    if is_async:
        tags.append("async")
    # purity / effects
    text = src
    def has(word: str) -> bool:
        return re.search(rf"\b{re.escape(word)}\b", text) is not None

    has_io = any(has(k) for k in IO_KEYWORDS)
    if has_io:
        tags.append("impure")
    else:
        tags.append("pure-ish")

    if any(has(k) for k in ("requests", "urllib", "httpx", "selenium")):
        tags.append("network")
    if any(has(k) for k in ("sqlalchemy", "sqlite3", "psycopg2")):
        tags.append("db")
    if any(has(k) for k in ("open", "Path", "os", "shutil")):
        tags.append("filesystem")
    if any(has(k) for k in ("print", "logging")):
        tags.append("logging")
    if any(has(k) for k in ("random",)):
        tags.append("randomness")
    if any(has(k) for k in ("time", "datetime")):
        tags.append("time")
    if any(has(k) for k in ("re",)):
        tags.append("regex")
    if any(has(k) for k in ("json",)):
        tags.append("json")

    # size bucket
    lines = src.splitlines()
    loc = len(lines)
    if loc < 10:
        tags.append("size:tiny")
    elif loc < 30:
        tags.append("size:small")
    elif loc < 100:
        tags.append("size:medium")
    else:
        tags.append("size:large")

    return tags


def _normalize_tokens(code: str) -> str:
    # Replace identifiers and literals with placeholders; keep punctuation/operators/keywords
    out: List[str] = []
    kw = set(keyword.kwlist)
    try:
        reader = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(reader):
            ttype, tstr = tok.type, tok.string
            if ttype == tokenize.NAME:
                if tstr in kw:
                    out.append(tstr)
                else:
                    out.append("NAME")
            elif ttype in (tokenize.STRING, tokenize.NUMBER):
                out.append("LIT")
            elif ttype in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT):
                continue
            else:
                # keep operators, punctuation
                s = tstr.strip()
                if s:
                    out.append(s)
    except Exception:
        return code
    return " ".join(out)


def _simhash64(tokens: Iterable[str]) -> int:
    # Simple 64-bit simhash
    v = [0] * 64
    for tok in tokens:
        h = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(64):
            bit = 1 if (h >> i) & 1 else -1
            v[i] += bit
    x = 0
    for i in range(64):
        if v[i] >= 0:
            x |= (1 << i)
    return x


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _signature_from_node(node: ast.AST) -> Tuple[str, int, bool, bool]:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return "()", 0, False, False
    a = node.args
    arg_names = [getattr(arg, "arg", "_") for arg in list(a.posonlyargs) + list(a.args)]
    if a.vararg:
        arg_names.append("*args")
    if a.kwonlyargs:
        arg_names.extend([getattr(k, "arg", "_") for k in a.kwonlyargs])
    if a.kwarg:
        arg_names.append("**kwargs")
    sig = f"({', '.join(arg_names)})"
    return sig, len(arg_names), bool(a.vararg), bool(a.kwarg)


class _FuncVisitor(ast.NodeVisitor):
    def __init__(self, module_path: str, source: str) -> None:
        self.module_path = module_path
        self.source = source
        self.stack: List[str] = []
        self.items: List[FunctionInfo] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:  # type: ignore[override]
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:  # type: ignore[override]
        self._capture(node, is_async=False)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:  # type: ignore[override]
        self._capture(node, is_async=True)
        self.generic_visit(node)

    def _capture(self, node: ast.AST, is_async: bool) -> None:
        try:
            start, end = getattr(node, "lineno", 0), _end_lineno(node)
            lines = self.source.splitlines()
            frag = "\n".join(lines[start - 1 : end]) if 1 <= start <= len(lines) else ""
            sig, argc, varg, kwarg = _signature_from_node(node)
            returns = sum(isinstance(n, ast.Return) for n in ast.walk(node))
            yields = sum(isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom) for n in ast.walk(node))
            cx = _rough_complexity(node)
            tags = _collect_tags(frag, node, is_async)
            norm = _normalize_tokens(frag)
            fph = hashlib.sha1(norm.encode("utf-8")).hexdigest()
            simh = _simhash64(norm.split())
            qual = (".".join(self.stack) + "." if self.stack else "") + getattr(node, "name", "<fn>")
            info = FunctionInfo(
                module_path=self.module_path,
                qualname=qual,
                lineno=start,
                end_lineno=end,
                is_method=bool(self.stack),
                is_async=is_async,
                args_count=argc,
                has_varargs=varg,
                has_kwargs=kwarg,
                returns_count=returns,
                yield_count=yields,
                complexity=cx,
                loc=max(0, end - start + 1),
                tags=tags,
                signature=sig,
                fingerprint=fph,
                simhash64=simh,
                normalized=norm,
            )
            self.items.append(info)
        except Exception:
            # best effort; skip broken nodes
            pass


class CodeSimilarityClassifier:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or Path()
        self.functions: List[FunctionInfo] = []
        self.similar_pairs: List[SimilarityPair] = []

    def _iter_py_files(self) -> Iterable[Path]:
        for p in self.root.rglob("*.py"):
            ps = str(p)
            if any(part in ps for part in IGNORED_PARTS):
                continue
            if any(ps.endswith(suf) or (suf in ps) for suf in SKIP_FILE_PATTERNS):
                continue
            yield p

    def scan(self) -> List[FunctionInfo]:
        items: List[FunctionInfo] = []
        for path in self._iter_py_files():
            text = _safe_read(path)
            if not text:
                continue
            try:
                tree = ast.parse(text)
            except Exception:
                continue
            vis = _FuncVisitor(str(path), text)
            vis.visit(tree)
            items.extend(vis.items)
        self.functions = items
        return items

    @staticmethod
    def _bin_key(fi: FunctionInfo) -> Tuple[str, str, bool]:
        # group by async + size + coarse purity bucket
        size = next((t for t in fi.tags if t.startswith("size:")), "size:unknown")
        purity = "impure" if "impure" in fi.tags else "pure"
        return size, purity, fi.is_async

    def _eligible(self, a: FunctionInfo, b: FunctionInfo) -> bool:
        if a is b:
            return False
        # Similar size (±30%)
        if a.loc == 0 or b.loc == 0:
            return False
        if not (0.7 <= a.loc / b.loc <= 1.3):
            return False
        # Avoid comparing a method to a top-level function unless sizes are tiny
        return not (a.is_method != b.is_method and not (a.loc < 15 and b.loc < 15))

    def find_similar(self, min_ratio: float = 0.88, max_hamming: int = 6) -> List[SimilarityPair]:
        pairs: List[SimilarityPair] = []
        if not self.functions:
            return pairs
        # index by bins
        bins: Dict[Tuple[str, str, bool], List[int]] = {}
        for i, f in enumerate(self.functions):
            bins.setdefault(self._bin_key(f), []).append(i)
        for _, idxs in bins.items():
            n = len(idxs)
            if n < 2:
                continue
            # Cap worst-case work
            if n > 600:
                idxs = idxs[:600]
                n = len(idxs)
            for i in range(n):
                a = self.functions[idxs[i]]
                for j in range(i + 1, n):
                    b = self.functions[idxs[j]]
                    if not self._eligible(a, b):
                        continue
                    ham = _hamming(a.simhash64, b.simhash64)
                    if ham > max_hamming:
                        continue
                    # quick name heuristic: ignore totally different verbs when tiny
                    name_sim = difflib.SequenceMatcher(None, a.qualname.split(".")[-1], b.qualname.split(".")[-1]).ratio()
                    # detailed ratio on normalized token streams
                    ratio = difflib.SequenceMatcher(None, a.normalized, b.normalized).ratio()
                    if ratio >= min_ratio:
                        reason = f"sequence_match={ratio:.2f}, simhash_hamming={ham}, name_sim={name_sim:.2f}"
                        pairs.append(SimilarityPair(a_idx=idxs[i], b_idx=idxs[j], score=ratio, hamming=ham, reason=reason))
        # sort high to low
        pairs.sort(key=lambda p: (p.score, -p.hamming), reverse=True)
        self.similar_pairs = pairs
        return pairs

    def clusters(self) -> List[List[int]]:
        # Build components over function indices using current similar_pairs
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            parent.setdefault(x, x)
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for sp in self.similar_pairs:
            union(sp.a_idx, sp.b_idx)
        groups: Dict[int, List[int]] = {}
        for idx in set([sp.a_idx for sp in self.similar_pairs] + [sp.b_idx for sp in self.similar_pairs]):
            r = find(idx)
            groups.setdefault(r, []).append(idx)
        return list(groups.values())

    def to_json(self) -> Dict[str, Any]:
        return {
            "functions": [asdict(f) for f in self.functions],
            "similar_pairs": [asdict(p) for p in self.similar_pairs],
            "clusters": self.clusters(),
        }


# ----------------------------- CLI / Runner --------------------------------

def _print_summary(clsfr: CodeSimilarityClassifier, top_n: int = 30) -> None:
    print("=== FUNCTION CLASSIFICATION REPORT START ===")
    print(f"Total functions/methods analyzed: {len(clsfr.functions)}")
    by_tag: Dict[str, int] = {}
    for f in clsfr.functions:
        for t in f.tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    top_tags = ", ".join(f"{k}:{v}" for k, v in sorted(by_tag.items(), key=lambda kv: kv[1], reverse=True)[:10])
    print(f"Top tags: {top_tags}")
    print(f"Similar pairs found: {len(clsfr.similar_pairs)} (showing top {min(top_n, len(clsfr.similar_pairs))})")
    for sp in clsfr.similar_pairs[:top_n]:
        a = clsfr.functions[sp.a_idx]
        b = clsfr.functions[sp.b_idx]
        print(f"- {Path(a.module_path).name}:{a.qualname} <-> {Path(b.module_path).name}:{b.qualname} | {sp.reason}")
    # Clusters
    comps = clsfr.clusters()
    if comps:
        print(f"Clusters: {len(comps)}")
        for c in comps[:10]:
            names = [clsfr.functions[i].qualname for i in c]
            print("  • " + ", ".join(names[:5]) + (" ..." if len(names) > 5 else ""))
    print("=== FUNCTION CLASSIFICATION REPORT END ===")


def main(argv: List[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Classify functions/methods and detect near-duplicates")
    ap.add_argument("--root", default=".", help="Root directory to scan")
    ap.add_argument("--min-similarity", type=float, default=0.88)
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--top", type=int, default=30, help="How many similar pairs to print")
    ap.add_argument("--save-json", type=str, default="", help="Optional path to write JSON report")
    args = ap.parse_args(argv)

    clsfr = CodeSimilarityClassifier(Path(args.root))
    clsfr.scan()
    clsfr.find_similar(min_ratio=args.min_similarity, max_hamming=args.max_hamming)
    _print_summary(clsfr, top_n=args.top)

    if args.save_json:
        try:
            out = Path(args.save_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(clsfr.to_json(), indent=2), encoding="utf-8")
            print(f"Saved JSON report to {out}")
        except Exception as e:
            print(f"Failed to save JSON report: {e}")
            return 1
    return 0


# ----------------------------- Tests (inline) -------------------------------

def run_comprehensive_tests() -> bool:
    """Minimal, strict tests that validate this module actually analyzes code."""
    try:
        clsfr = CodeSimilarityClassifier(Path())
        funcs = clsfr.scan()
        # Expect a nontrivial codebase: require at least 40 functions discovered
        assert len(funcs) >= 40, f"Too few functions discovered: {len(funcs)}"
        # Sanity check of fields
        sample = funcs[min(5, len(funcs) - 1)]
        assert sample.module_path.endswith(".py"), "module_path should be a .py file"
        assert sample.qualname and isinstance(sample.qualname, str)
        assert sample.loc >= 1 and sample.lineno >= 1
        # Similarity should find at least a few candidates in a real repo
        pairs = clsfr.find_similar(min_ratio=0.86, max_hamming=10)
        # Not required to be many, but expect at least 1 in a sizable repo
        assert len(pairs) >= 1, "No similar function pairs found — unexpected in this repo"
        # JSON serialization
        blob = clsfr.to_json()
        assert "functions" in blob and "similar_pairs" in blob and "clusters" in blob
        return True
    except AssertionError as e:
        print(f"TEST FAILURE (code_similarity_classifier): {e}")
        return False
    except Exception as e:
        print(f"TEST ERROR (code_similarity_classifier): {e}")
        return False


if __name__ == "__main__":
    sys.exit(main())

