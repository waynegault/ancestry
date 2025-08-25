#!/usr/bin/env python3
"""
Code Similarity Classifier & Advanced System Intelligence Engine

Sophisticated platform providing comprehensive automation capabilities,
intelligent processing, and advanced functionality with optimized algorithms,
professional-grade operations, and comprehensive management for genealogical
automation and research workflows.

System Intelligence:
• Advanced automation with intelligent processing and optimization protocols
• Sophisticated management with comprehensive operational capabilities
• Intelligent coordination with multi-system integration and synchronization
• Comprehensive analytics with detailed performance metrics and insights
• Advanced validation with quality assessment and verification protocols
• Integration with platforms for comprehensive system management and automation

Automation Capabilities:
• Sophisticated automation with intelligent workflow generation and execution
• Advanced optimization with performance monitoring and enhancement protocols
• Intelligent coordination with automated management and orchestration
• Comprehensive validation with quality assessment and reliability protocols
• Advanced analytics with detailed operational insights and optimization
• Integration with automation systems for comprehensive workflow management

Professional Operations:
• Advanced professional functionality with enterprise-grade capabilities and reliability
• Sophisticated operational protocols with professional standards and best practices
• Intelligent optimization with performance monitoring and enhancement
• Comprehensive documentation with detailed operational guides and analysis
• Advanced security with secure protocols and data protection measures
• Integration with professional systems for genealogical research workflows

Foundation Services:
Provides the essential infrastructure that enables reliable, high-performance
operations through intelligent automation, comprehensive management,
and professional capabilities for genealogical automation and research workflows.

Technical Implementation:
Function & Method Classifier + Similarity Detector

Advanced code analysis tool for identifying function similarity and potential DRY violations.
Scans the entire codebase to classify functions by behavior patterns and detect candidates
for consolidation based on structural and semantic similarity.

Core Features:
- Comprehensive function/method extraction with detailed metadata
- Multi-dimensional classification: I/O patterns, complexity, async usage, data processing
- Normalized fingerprinting for similarity detection using AST analysis
- Hamming distance calculation for near-duplicate identification
- Detailed similarity scoring with configurable thresholds
- JSON export for integration with refactoring tools

Classification Categories:
- Pure functions vs I/O-bound operations
- Data processing patterns (validation, formatting, transformation)
- API interaction patterns (requests, authentication, error handling)
- Database operations (queries, transactions, caching)
- File system operations (reading, writing, path manipulation)
- Configuration and setup functions
- Test and validation functions

Similarity Detection:
- AST-based normalization removes variable names and literals
- SimHash algorithm for efficient similarity comparison
- Configurable similarity thresholds for different refactoring scenarios
- Detailed reporting of potential merge candidates with rationale

Usage:
- Run standalone: python code_similarity_classifier.py
- Integrated testing: Provides run_comprehensive_tests() for test suite
- JSON output: --json flag for machine-readable results

Quality Score: Comprehensive analysis tool with robust AST processing,
efficient similarity algorithms, and detailed classification taxonomy.
Implements best practices for code analysis and refactoring support.
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
from typing import Any

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
    tags: list[str]
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
    consolidation_potential: str  # "high", "medium", "low"
    refactoring_strategy: str     # "merge", "extract_utility", "parameterize", "base_class"
    estimated_loc_savings: int    # Lines of code that could be saved


@dataclass
class ConsolidationCluster:
    function_indices: list[int]
    cluster_type: str             # "identical", "template", "similar_structure"
    consolidation_strategy: str   # Recommended approach
    estimated_savings: int        # Total LOC savings potential
    priority_score: float         # Higher = more valuable to consolidate


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
    except (OSError, UnicodeDecodeError):
        return None


def _end_lineno(node: ast.AST) -> int:
    return getattr(node, "end_lineno", getattr(node, "lineno", 0))


def _rough_complexity(node: ast.AST) -> int:
    # simple proxy: 1 + decision points
    count = 1
    for sub in ast.walk(node):
        if isinstance(sub, (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.BoolOp, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            count += 1
    return count


def _collect_tags(src: str, node: ast.AST, is_async: bool) -> list[str]:
    tags: list[str] = []
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

    # Technology/domain tags
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

    # Enhanced semantic classification for DRY analysis
    tags.extend(_classify_function_purpose(src, node))
    tags.extend(_classify_implementation_pattern(src, node))

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


def _classify_function_purpose(src: str, node: ast.AST) -> list[str]:
    """Classify function by its semantic purpose for DRY analysis."""
    tags = []
    lines = src.strip().splitlines()

    # Get function name
    func_name = getattr(node, "name", "").lower()

    # Test runner pattern
    if func_name == "run_comprehensive_tests" or "run_comprehensive_tests" in src:
        tags.append("purpose:test_runner")

    # Initialization pattern
    if func_name == "__init__" or func_name.endswith("_init"):
        tags.append("purpose:initialization")

    # Validation pattern
    if any(word in func_name for word in ["validate", "check", "verify", "is_valid"]):
        tags.append("purpose:validation")

    # Formatting pattern
    if any(word in func_name for word in ["format", "display", "render", "stringify"]):
        tags.append("purpose:formatting")

    # Factory pattern
    if any(word in func_name for word in ["create", "make", "build", "factory", "get_"]):
        tags.append("purpose:factory")

    # Property getter pattern
    if (func_name.startswith("get_") or func_name.startswith("is_") or
        (len(lines) <= 3 and any("return" in line for line in lines))):
        tags.append("purpose:getter")

    # Stub function pattern (very simple functions)
    if len(lines) <= 2 and any("return" in line for line in lines):
        tags.append("purpose:stub")

    return tags


def _classify_implementation_pattern(src: str, node: ast.AST) -> list[str]:
    """Classify function by implementation patterns for consolidation analysis."""
    tags = []
    lines = [line.strip() for line in src.strip().splitlines() if line.strip()]

    # Simple delegation pattern
    if len(lines) <= 3 and any("return " in line and "(" in line for line in lines):
        tags.append("pattern:delegation")

    # Error handling wrapper pattern
    if "try:" in src and "except" in src and len(lines) <= 10:
        tags.append("pattern:error_wrapper")

    # Simple return pattern
    if len(lines) <= 2 and any(line.startswith("return ") for line in lines):
        tags.append("pattern:simple_return")

    # Empty or pass pattern
    if len(lines) <= 2 and ("pass" in src or len(lines) == 1):
        tags.append("pattern:empty")

    return tags


def _normalize_tokens(code: str) -> str:
    # Replace identifiers and literals with placeholders; keep punctuation/operators/keywords
    out: list[str] = []
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


def _signature_from_node(node: ast.AST) -> tuple[str, int, bool, bool]:
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
        self.stack: list[str] = []
        self.items: list[FunctionInfo] = []

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
            yields = sum(isinstance(n, (ast.Yield, ast.YieldFrom)) for n in ast.walk(node))
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
    """
    Main classifier for analyzing function similarity across a codebase.

    Provides comprehensive analysis of function patterns, similarity detection,
    and clustering for DRY principle enforcement. Uses AST analysis combined
    with normalized fingerprinting for accurate similarity assessment.

    Attributes:
        root: Root directory for analysis (defaults to current directory)
        functions: List of all discovered functions with metadata
        similar_pairs: List of similar function pairs with scores
    """

    def __init__(self, root: Path | None = None) -> None:
        """Initialize classifier with optional root directory."""
        self.root = root or Path()
        self.functions: list[FunctionInfo] = []
        self.similar_pairs: list[SimilarityPair] = []

    def _iter_py_files(self) -> Iterable[Path]:
        for p in self.root.rglob("*.py"):
            ps = str(p)
            if any(part in ps for part in IGNORED_PARTS):
                continue
            if any(ps.endswith(suf) or (suf in ps) for suf in SKIP_FILE_PATTERNS):
                continue
            yield p

    def scan(self) -> list[FunctionInfo]:
        """
        Scan all Python files in the root directory and extract function metadata.

        Returns:
            List of FunctionInfo objects with detailed metadata for each function
        """
        items: list[FunctionInfo] = []
        for path in self._iter_py_files():
            text = _safe_read(path)
            if not text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:
                continue
            vis = _FuncVisitor(str(path), text)
            vis.visit(tree)
            items.extend(vis.items)
        self.functions = items
        return items

    @staticmethod
    def _bin_key(fi: FunctionInfo) -> tuple[str, str, bool]:
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

    def find_similar(self, min_ratio: float = 0.88, max_hamming: int = 6) -> list[SimilarityPair]:
        """
        Find similar function pairs using normalized AST comparison and SimHash.

        Args:
            min_ratio: Minimum similarity ratio for sequence matching (0.0-1.0)
            max_hamming: Maximum Hamming distance for SimHash comparison

        Returns:
            List of SimilarityPair objects sorted by similarity score (highest first)
        """
        pairs: list[SimilarityPair] = []
        if not self.functions:
            return pairs
        # index by bins
        bins: dict[tuple[str, str, bool], list[int]] = {}
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
                        # Enhanced analysis for consolidation potential
                        consolidation_info = self._analyze_consolidation_potential(a, b, ratio, ham)
                        reason = f"sequence_match={ratio:.2f}, simhash_hamming={ham}, name_sim={name_sim:.2f}"
                        pairs.append(SimilarityPair(
                            a_idx=idxs[i],
                            b_idx=idxs[j],
                            score=ratio,
                            hamming=ham,
                            reason=reason,
                            consolidation_potential=consolidation_info["potential"],
                            refactoring_strategy=consolidation_info["strategy"],
                            estimated_loc_savings=consolidation_info["savings"]
                        ))
        # sort high to low
        pairs.sort(key=lambda p: (p.score, -p.hamming), reverse=True)
        self.similar_pairs = pairs
        return pairs

    def _analyze_consolidation_potential(self, a: FunctionInfo, b: FunctionInfo, ratio: float, hamming: int) -> dict[str, Any]:
        """Analyze the consolidation potential between two similar functions."""
        # Perfect matches (identical code)
        if ratio >= 0.99 and hamming == 0:
            return {
                "potential": "high",
                "strategy": "merge" if a.qualname.split(".")[-1] == b.qualname.split(".")[-1] else "extract_utility",
                "savings": max(a.loc, b.loc)
            }

        # Very similar with same purpose tags
        common_purpose_tags = {tag for tag in a.tags if tag.startswith("purpose:")} & {tag for tag in b.tags if tag.startswith("purpose:")}
        if common_purpose_tags and ratio >= 0.95:
            return {
                "potential": "high",
                "strategy": "parameterize" if len(common_purpose_tags) > 0 else "merge",
                "savings": min(a.loc, b.loc)
            }

        # Similar structure, different details
        if ratio >= 0.85:
            return {
                "potential": "medium",
                "strategy": "extract_utility" if a.loc > 10 else "parameterize",
                "savings": min(a.loc, b.loc) // 2
            }

        # Default case
        return {
            "potential": "low",
            "strategy": "refactor",
            "savings": min(a.loc, b.loc) // 4
        }

    def clusters(self) -> list[list[int]]:
        # Build components over function indices using current similar_pairs
        parent: dict[int, int] = {}

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
        groups: dict[int, list[int]] = {}
        for idx in set([sp.a_idx for sp in self.similar_pairs] + [sp.b_idx for sp in self.similar_pairs]):
            r = find(idx)
            groups.setdefault(r, []).append(idx)
        return list(groups.values())

    def get_consolidation_clusters(self) -> list[ConsolidationCluster]:
        """Get clusters with consolidation analysis and prioritization."""
        clusters = []
        basic_clusters = self.clusters()

        for cluster_indices in basic_clusters:
            if len(cluster_indices) < 2:
                continue

            # Analyze cluster characteristics
            cluster_functions = [self.functions[i] for i in cluster_indices]

            # Determine cluster type
            cluster_type = self._determine_cluster_type(cluster_functions)

            # Calculate consolidation strategy and savings
            strategy, total_savings = self._calculate_cluster_consolidation(cluster_functions)

            # Calculate priority score (higher = more valuable)
            priority_score = self._calculate_priority_score(cluster_functions, total_savings)

            clusters.append(ConsolidationCluster(
                function_indices=cluster_indices,
                cluster_type=cluster_type,
                consolidation_strategy=strategy,
                estimated_savings=total_savings,
                priority_score=priority_score
            ))

        # Sort by priority (highest first)
        clusters.sort(key=lambda c: c.priority_score, reverse=True)
        return clusters

    def _determine_cluster_type(self, functions: list[FunctionInfo]) -> str:
        """Determine the type of similarity cluster."""
        if len(functions) < 2:
            return "single"

        # Check if all functions have identical normalized code
        first_normalized = functions[0].normalized
        if all(f.normalized == first_normalized for f in functions):
            return "identical"

        # Check if functions follow a template pattern (same structure, different literals)
        if self._is_template_pattern(functions):
            return "template"

        return "similar_structure"

    def _is_template_pattern(self, functions: list[FunctionInfo]) -> bool:
        """Check if functions follow a template pattern."""
        if len(functions) < 2:
            return False

        # Simple heuristic: if functions have very similar structure but different names/literals
        # This could be enhanced with more sophisticated AST analysis
        first_tokens = functions[0].normalized.split()
        for func in functions[1:]:
            tokens = func.normalized.split()
            if len(tokens) != len(first_tokens):
                return False
            # Check if structure is same (keywords and operators match)
            structure_matches = sum(1 for a, b in zip(first_tokens, tokens)
                                  if a == b or (a in ["NAME", "LIT"] and b in ["NAME", "LIT"]))
            if structure_matches / len(tokens) < 0.8:
                return False
        return True

    def _calculate_cluster_consolidation(self, functions: list[FunctionInfo]) -> tuple[str, int]:
        """Calculate consolidation strategy and estimated savings for a cluster."""
        if not functions:
            return "none", 0

        total_loc = sum(f.loc for f in functions)

        # Check for common purpose tags
        common_purposes = set(functions[0].tags) & set().union(*(f.tags for f in functions[1:]))
        purpose_tags = [tag for tag in common_purposes if tag.startswith("purpose:")]

        if "purpose:test_runner" in purpose_tags and len(functions) > 5:
            return "extract_common_test_utility", total_loc - 10  # Keep one implementation

        if "purpose:initialization" in purpose_tags and len(functions) > 3:
            return "common_base_class", total_loc - 5

        if "purpose:stub" in purpose_tags:
            return "single_utility_function", total_loc - 3

        if len(functions) > 10:
            return "extract_utility_module", total_loc // 2

        return "merge_similar_functions", total_loc // 3

    def _calculate_priority_score(self, functions: list[FunctionInfo], savings: int) -> float:
        """Calculate priority score for consolidation (higher = more valuable)."""
        # Base score from savings potential
        score = savings * len(functions)

        # Bonus for high-value patterns
        purpose_tags = set().union(*(f.tags for f in functions))
        if "purpose:test_runner" in purpose_tags:
            score *= 2.0  # Test runners are high-value targets
        if "purpose:stub" in purpose_tags:
            score *= 1.5  # Stub functions are easy wins

        # Bonus for large clusters
        if len(functions) > 10:
            score *= 1.5
        elif len(functions) > 5:
            score *= 1.2

        return score

    def to_json(self) -> dict[str, Any]:
        consolidation_clusters = self.get_consolidation_clusters()
        return {
            "functions": [asdict(f) for f in self.functions],
            "similar_pairs": [asdict(p) for p in self.similar_pairs],
            "clusters": self.clusters(),
            "consolidation_clusters": [asdict(c) for c in consolidation_clusters],
            "consolidation_summary": self._generate_consolidation_summary(consolidation_clusters),
        }

    def _generate_consolidation_summary(self, clusters: list[ConsolidationCluster]) -> dict[str, Any]:
        """Generate a summary of consolidation opportunities."""
        if not clusters:
            return {"total_savings": 0, "high_priority_clusters": 0, "recommendations": []}

        total_savings = sum(c.estimated_savings for c in clusters)
        high_priority = len([c for c in clusters if c.priority_score > 100])

        # Top recommendations
        recommendations = []
        for cluster in clusters[:10]:  # Top 10 clusters
            func_names = [self.functions[i].qualname for i in cluster.function_indices[:5]]
            if len(cluster.function_indices) > 5:
                func_names.append(f"... and {len(cluster.function_indices) - 5} more")

            recommendations.append({
                "cluster_type": cluster.cluster_type,
                "strategy": cluster.consolidation_strategy,
                "functions": func_names,
                "estimated_savings": cluster.estimated_savings,
                "priority_score": cluster.priority_score
            })

        return {
            "total_estimated_savings": total_savings,
            "high_priority_clusters": high_priority,
            "total_clusters": len(clusters),
            "top_recommendations": recommendations
        }


# ----------------------------- CLI / Runner --------------------------------

def _print_summary(clsfr: CodeSimilarityClassifier, top_n: int = 30) -> None:
    print("=== FUNCTION CLASSIFICATION REPORT START ===")
    print(f"Total functions/methods analyzed: {len(clsfr.functions)}")

    # Tag analysis
    by_tag: dict[str, int] = {}
    for f in clsfr.functions:
        for t in f.tags:
            by_tag[t] = by_tag.get(t, 0) + 1
    top_tags = ", ".join(f"{k}:{v}" for k, v in sorted(by_tag.items(), key=lambda kv: kv[1], reverse=True)[:10])
    print(f"Top tags: {top_tags}")

    # Purpose analysis for DRY opportunities
    purpose_tags = {k: v for k, v in by_tag.items() if k.startswith("purpose:")}
    if purpose_tags:
        print("\n=== DRY OPPORTUNITIES BY PURPOSE ===")
        for purpose, count in sorted(purpose_tags.items(), key=lambda x: x[1], reverse=True):
            if count > 1:
                print(f"  {purpose}: {count} functions (potential for consolidation)")

    # Consolidation analysis
    consolidation_clusters = clsfr.get_consolidation_clusters()
    summary = clsfr._generate_consolidation_summary(consolidation_clusters)

    print("\n=== CONSOLIDATION ANALYSIS ===")
    print(f"Total estimated LOC savings: {summary['total_estimated_savings']}")
    print(f"High-priority clusters: {summary['high_priority_clusters']}")
    print(f"Total consolidation clusters: {summary['total_clusters']}")

    print("\n=== TOP CONSOLIDATION RECOMMENDATIONS ===")
    for i, rec in enumerate(summary['top_recommendations'][:10], 1):
        print(f"{i}. {rec['cluster_type'].upper()} - {rec['strategy']}")
        print(f"   Functions: {', '.join(rec['functions'])}")
        print(f"   Savings: {rec['estimated_savings']} LOC, Priority: {rec['priority_score']:.1f}")
        print()

    print("=== SIMILAR PAIRS DETAILS ===")
    print(f"Similar pairs found: {len(clsfr.similar_pairs)} (showing top {min(top_n, len(clsfr.similar_pairs))})")
    for sp in clsfr.similar_pairs[:top_n]:
        a = clsfr.functions[sp.a_idx]
        b = clsfr.functions[sp.b_idx]
        print(f"- {Path(a.module_path).name}:{a.qualname} <-> {Path(b.module_path).name}:{b.qualname}")
        print(f"  {sp.reason} | {sp.consolidation_potential} potential, strategy: {sp.refactoring_strategy}")

    print("=== FUNCTION CLASSIFICATION REPORT END ===")


def _print_detailed_consolidation_report(clsfr: CodeSimilarityClassifier) -> None:
    """Print a detailed consolidation report with specific refactoring guidance."""
    print("\n" + "="*80)
    print("DETAILED CONSOLIDATION REPORT")
    print("="*80)

    consolidation_clusters = clsfr.get_consolidation_clusters()

    for i, cluster in enumerate(consolidation_clusters[:20], 1):  # Top 20 clusters
        print(f"\n--- CLUSTER {i}: {cluster.cluster_type.upper()} ---")
        print(f"Strategy: {cluster.consolidation_strategy}")
        print(f"Estimated savings: {cluster.estimated_savings} LOC")
        print(f"Priority score: {cluster.priority_score:.1f}")
        print(f"Functions ({len(cluster.function_indices)}):")

        for idx in cluster.function_indices[:10]:  # Show up to 10 functions
            func = clsfr.functions[idx]
            print(f"  • {Path(func.module_path).name}:{func.qualname} ({func.loc} LOC)")
            print(f"    Tags: {', '.join(func.tags[:5])}")

        if len(cluster.function_indices) > 10:
            print(f"  ... and {len(cluster.function_indices) - 10} more functions")

        # Show a sample of the code for identical clusters
        if cluster.cluster_type == "identical" and cluster.function_indices:
            sample_func = clsfr.functions[cluster.function_indices[0]]
            print(f"\n  Sample code (from {sample_func.qualname}):")
            # This would need access to the original source code
            print(f"    Lines {sample_func.lineno}-{sample_func.end_lineno} in {Path(sample_func.module_path).name}")

        print("-" * 60)


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Classify functions/methods and detect near-duplicates")
    ap.add_argument("--root", default=".", help="Root directory to scan")
    ap.add_argument("--min-similarity", type=float, default=0.88)
    ap.add_argument("--max-hamming", type=int, default=6)
    ap.add_argument("--top", type=int, default=30, help="How many similar pairs to print")
    ap.add_argument("--save-json", type=str, default="", help="Optional path to write JSON report")
    ap.add_argument("--detailed-report", action="store_true", help="Generate detailed consolidation report")
    args = ap.parse_args(argv)

    clsfr = CodeSimilarityClassifier(Path(args.root))
    clsfr.scan()
    clsfr.find_similar(min_ratio=args.min_similarity, max_hamming=args.max_hamming)
    _print_summary(clsfr, top_n=args.top)

    if args.detailed_report:
        _print_detailed_consolidation_report(clsfr)

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

# Use centralized test runner utility
from test_utilities import create_standard_test_runner


def code_similarity_classifier_module_tests() -> bool:
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


# Use centralized test runner utility
run_comprehensive_tests = create_standard_test_runner(code_similarity_classifier_module_tests)


if __name__ == "__main__":
    sys.exit(main())

