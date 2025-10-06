#!/usr/bin/env python3
"""
Automated refactoring tool for too-many-arguments violations.

This script analyzes functions with >5 parameters and suggests/applies
refactorings using dataclasses to group related parameters.
"""

import ast
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class FunctionInfo:
    """Information about a function with too many arguments."""
    file_path: str
    line_number: int
    function_name: str
    parameters: List[Tuple[str, Optional[str]]]  # (name, type_annotation)
    parameter_count: int


@dataclass
class ParameterGroup:
    """A group of related parameters that could become a dataclass."""
    name: str
    parameters: List[Tuple[str, Optional[str]]]
    score: float  # How confident we are this is a good grouping


def get_violations() -> List[Dict[str, Any]]:
    """Get all PLR0913 violations from ruff."""
    result = subprocess.run(
        ['python', '-m', 'ruff', 'check', '--select=PLR0913', '.', '--output-format=json'],
        capture_output=True,
        text=True, check=False
    )
    if result.stdout:
        return json.loads(result.stdout)
    return []


def parse_function_signature(file_path: str, line_number: int) -> Optional[FunctionInfo]:
    """Parse a function signature to extract parameter information."""
    try:
        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.lineno == line_number:
                params = []
                for arg in node.args.args:
                    type_ann = None
                    if arg.annotation:
                        type_ann = ast.unparse(arg.annotation)
                    params.append((arg.arg, type_ann))

                return FunctionInfo(
                    file_path=file_path,
                    line_number=line_number,
                    function_name=node.name,
                    parameters=params,
                    parameter_count=len(params)
                )
    except Exception as e:
        print(f"Error parsing {file_path}:{line_number}: {e}")
    return None


def identify_parameter_patterns(functions: List[FunctionInfo]) -> Dict[str, List[str]]:
    """Identify common parameter name patterns across functions."""
    param_frequency = defaultdict(int)
    param_types = defaultdict(set)

    for func in functions:
        for param_name, param_type in func.parameters:
            param_frequency[param_name] += 1
            if param_type:
                param_types[param_name].add(param_type)

    # Group parameters by common patterns
    patterns = {
        'counters': ['new', 'updated', 'skipped', 'errors', 'count', 'total'],
        'identifiers': ['uuid', 'id', 'username', 'name', 'ref'],
        'session': ['session', 'session_manager', 'driver', 'db_session'],
        'config': ['config', 'settings', 'options', 'params'],
        'logging': ['logger', 'log_ref', 'logger_instance'],
        'data': ['match', 'person', 'tree', 'data', 'record'],
        'flags': ['in_my_tree', 'is_valid', 'should_update', 'needs_update'],
    }

    return patterns


def suggest_parameter_groupings(func_info: FunctionInfo) -> List[ParameterGroup]:
    """Suggest how to group parameters for a specific function."""
    patterns = {
        'counters': ['new', 'updated', 'skipped', 'errors', 'count', 'total'],
        'identifiers': ['uuid', 'id', 'username', 'name', 'ref'],
        'session': ['session', 'session_manager', 'driver', 'db_session'],
        'config': ['config', 'settings', 'options', 'params'],
        'logging': ['logger', 'log_ref', 'logger_instance'],
        'data': ['match', 'person', 'tree', 'data', 'record'],
        'flags': ['in_my_tree', 'is_valid', 'should_update', 'needs_update'],
    }

    suggestions = []
    param_dict = {name: (name, type_ann) for name, type_ann in func_info.parameters}

    for pattern_name, keywords in patterns.items():
        matching_params = []
        for param_name, param_info in param_dict.items():
            if any(keyword in param_name.lower() for keyword in keywords):
                matching_params.append(param_info)

        if len(matching_params) >= 2:  # Only suggest if we can group 2+ params
            score = len(matching_params) / func_info.parameter_count
            suggestions.append(ParameterGroup(
                name=pattern_name,
                parameters=matching_params,
                score=score
            ))

    # Sort by score (best suggestions first)
    suggestions.sort(key=lambda x: x.score, reverse=True)
    return suggestions


def generate_dataclass_code(group: ParameterGroup, class_name: str) -> str:
    """Generate dataclass code for a parameter group."""
    lines = [
        "@dataclass",
        f"class {class_name}:",
        f'    """Groups {group.name}-related parameters."""'
    ]

    for param_name, param_type in group.parameters:
        if param_type:
            lines.append(f"    {param_name}: {param_type}")
        else:
            lines.append(f"    {param_name}: Any")

    return "\n".join(lines)


def analyze_and_report():
    """Analyze all violations and generate a report."""
    print("🔍 Analyzing too-many-arguments violations...\n")

    violations = get_violations()
    print(f"Found {len(violations)} violations\n")

    # Parse function signatures
    functions = []
    for violation in violations:
        file_path = violation['filename']
        line_number = violation['location']['row']
        func_info = parse_function_signature(file_path, line_number)
        if func_info:
            functions.append(func_info)

    print(f"Successfully parsed {len(functions)} function signatures\n")

    # Group by file
    by_file = defaultdict(list)
    for func in functions:
        by_file[func.file_path].append(func)

    # Generate report
    print("=" * 80)
    print("REFACTORING SUGGESTIONS")
    print("=" * 80)

    for file_path, funcs in sorted(by_file.items(), key=lambda x: -len(x[1])):
        print(f"\n📁 {Path(file_path).name} ({len(funcs)} violations)")
        print("-" * 80)

        for func in funcs[:3]:  # Show top 3 per file
            print(f"\n  Function: {func.function_name} ({func.parameter_count} params)")
            print(f"  Line: {func.line_number}")

            suggestions = suggest_parameter_groupings(func)
            if suggestions:
                best = suggestions[0]
                print(f"\n  💡 Suggestion: Group {len(best.parameters)} params into '{best.name}' dataclass")
                print(f"     Parameters: {', '.join(p[0] for p in best.parameters)}")
                print(f"     Confidence: {best.score:.0%}")
            else:
                print("  ⚠️  No automatic grouping suggestion")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total violations: {len(violations)}")
    print(f"Files affected: {len(by_file)}")
    print(f"Average params per function: {sum(f.parameter_count for f in functions) / len(functions):.1f}")

    # Identify most common parameter names
    param_freq = defaultdict(int)
    for func in functions:
        for param_name, _ in func.parameters:
            param_freq[param_name] += 1

    print("\nMost common parameter names:")
    for param, count in sorted(param_freq.items(), key=lambda x: -x[1])[:10]:
        print(f"  {param}: {count} occurrences")


if __name__ == "__main__":
    analyze_and_report()

