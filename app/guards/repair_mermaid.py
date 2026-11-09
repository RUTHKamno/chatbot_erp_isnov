"""Réparation des diagrammes Mermaid (graph / flowchart) générés par un LLM.

Usage :
    llm_response = repair_mermaid_code(llm_response)

- Fonctionne sur la réponse COMPLÈTE du LLM (texte + blocs ```mermaid) : le texte autour est conservé.
- Idempotent : un diagramme déjà correct n'est pas modifié (hors indentation).
- Les autres types de diagrammes (sequenceDiagram, classDiagram...) ne sont pas touchés, hormis la fermeture du bloc.
"""

import re
from typing import List

_FENCE_RE = re.compile(
    r"```[ \t]*mermaid[ \t]*\r?\n(?P<code>.*?)(?:\r?\n[ \t]*```|\Z)",  # \Z : bloc non fermé (réponse tronquée)
    re.DOTALL | re.IGNORECASE,
)
_FLOW_HEADER_RE = re.compile(
    r"^(graph|flowchart)\b[ \t]*(TD|TB|BT|RL|LR)?[ \t]*;?[ \t]*$", re.IGNORECASE
)
_BARE_FLOW_RE = re.compile(r"^\s*(graph|flowchart)\b", re.IGNORECASE)
_OTHER_DIAGRAMS = (
    "sequencediagram",
    "classdiagram",
    "statediagram",
    "erdiagram",
    "gantt",
    "pie",
    "journey",
    "mindmap",
    "timeline",
    "gitgraph",
    "quadrantchart",
    "requirementdiagram",
    "c4context",
)
_PASSTHROUGH = (
    "classdef ",
    "class ",
    "style ",
    "linkstyle ",
    "click ",
    "%%",
    "direction ",
)

_CLOSERS = {
    "((": "))",
    "([": "])",
    "[[": "]]",
    "{{": "}}",
    "[": "]",
    "(": ")",
    "{": "}",
}
_NODE_START = re.compile(r"(?<![\w\"'])([A-Za-z_]\w*)(\(\(|\(\[|\[\[|\{\{|\[|\(|\{)")
_PLACEHOLDER = re.compile(r"\x00(\d+)\x00")


def _clean_label(label: str) -> str:
    label = label.strip()
    if len(label) >= 2 and label[0] == '"' and label[-1] == '"':
        label = label[1:-1].strip()
    return label.replace('"', "'")


def _quote_node_labels(line: str) -> str:
    """A[Texte (x)] -> A["Texte (x)"] ; gère (), {}, ((..)), ([..]), [[..]], {{..}}."""
    out: List[str] = []
    pos = 0
    while True:
        m = _NODE_START.search(line, pos)
        if not m:
            out.append(line[pos:])
            break
        node_id, opener = m.group(1), m.group(2)
        closer = _CLOSERS[opener]
        end_re = re.compile(re.escape(closer) + r"(?=[ \t]|;|&|-|=|$)")
        e = end_re.search(line, m.end())
        if not e:  # délimiteur fermant introuvable : on laisse tel quel
            out.append(line[pos : m.end()])
            pos = m.end()
            continue
        out.append(line[pos : m.start()])
        label = _clean_label(line[m.end() : e.start()]) or node_id
        out.append(f'{node_id}{opener}"{label}"{closer}')
        pos = e.end()
    return "".join(out)


def _repair_edge_line(line: str, split_chains: bool) -> str:
    strings: List[str] = []

    def mask(text: str) -> str:
        strings.append(text)
        return f"\x00{len(strings) - 1}\x00"

    # 1. Flèches / libellés d'arêtes
    line = re.sub(
        r"(-->|---|==>|-\.->)[ \t]*\|([^|]*)\|>", r"\1|\2|", line
    )  # -->|Texte|>  =>  -->|Texte|
    line = re.sub(
        r"--[ \t]+(?!>)([^|>]+?)[ \t]+-->", r"-->|\1|", line
    )  # -- Texte -->  =>  -->|Texte|
    line = re.sub(
        r"(-->|---|==>|-\.->)([ \t]*)\|([^|]*)\|",
        lambda m: (
            f"{m.group(1)}|{mask(chr(34) + _clean_label(m.group(3)) + chr(34))}|"
            if m.group(3).strip()
            else m.group(1)
        ),
        line,
    )

    # 2. Libellés de nœuds entre guillemets, puis protection de toutes les chaînes
    line = _quote_node_labels(line)
    line = re.sub(r'"[^"\n]*"', lambda m: mask(m.group(0)), line)

    # 3. Flèches simples / longues, identifiant réservé `end`
    line = re.sub(r"-{3,}>", "-->", line)  # --->  =>  -->
    line = re.sub(r"(?<![-=.])->(?!>)", "-->", line)  # ->    =>  -->
    line = re.sub(r"(?<![\w\x00])end(?![\w\x00])", "end_", line)  # `end` casse Mermaid

    # 4. Enchaînements A --> B --> C  =>  une transition par ligne
    parts = re.split(r"[ \t]*-->[ \t]*(?:\|([^|]*)\|)?[ \t]*", line)
    nodes, labels = parts[0::2], parts[1::2]
    if len(nodes) > 1 and all(n.strip() for n in nodes):
        if split_chains:
            lines = []
            for i, lab in enumerate(labels):
                arrow = f"-->|{lab}|" if lab else "-->"
                lines.append(f"{nodes[i].strip()} {arrow} {nodes[i + 1].strip()}")
            line = "\n    ".join(lines)
        else:
            line = nodes[0].strip()
            for lab, nxt in zip(labels, nodes[1:]):
                line += f" -->|{lab}| " + nxt.strip() if lab else " --> " + nxt.strip()

    # 5. Restauration des chaînes protégées
    return _PLACEHOLDER.sub(lambda m: strings[int(m.group(1))], line)


def _repair_block(code: str, split_chains: bool = True) -> str:
    code = (
        (code or "")
        .replace("\r\n", "\n")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .strip()
    )
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if not lines:
        return code
    if lines[0].lower().startswith(_OTHER_DIAGRAMS):
        return code  # autre type de diagramme : on ne touche pas

    hm = _FLOW_HEADER_RE.match(lines[0])
    if hm:
        header, body = (
            f"{hm.group(1).lower()} {(hm.group(2) or 'TD').upper()}",
            lines[1:],
        )
    else:  # en-tête manquant ou invalide
        header, body = "graph TD", lines

    out: List[str] = []
    sg = 0
    for line in body:
        low = line.lower()
        if low == "end" or low.startswith(_PASSTHROUGH):
            out.append("    " + line)
        elif low.startswith("subgraph"):
            title = line[len("subgraph") :].strip()
            if (
                title and "[" not in title
            ):  # subgraph Étape 1 : Création  =>  subgraph sg1["Étape 1 : Création"]
                sg += 1
                line = f'subgraph sg{sg}["{_clean_label(title)}"]'
            out.append("    " + line)
        else:
            out.append("    " + _repair_edge_line(line.rstrip(";"), split_chains))
    return header + "\n" + "\n".join(out)


def repair_mermaid_code(text: str, split_chains: bool = True) -> str:
    """Répare tous les blocs ```mermaid d'une réponse LLM (ou un code Mermaid brut) et renvoie le texte complet."""
    if not text:
        return text
    if _FENCE_RE.search(text):
        return _FENCE_RE.sub(
            lambda m: "```mermaid\n"
            + _repair_block(m.group("code"), split_chains)
            + "\n```",
            text,
        )

    if _BARE_FLOW_RE.match(text):
        return "```mermaid\n" + _repair_block(text, split_chains) + "\n```"
    return text
