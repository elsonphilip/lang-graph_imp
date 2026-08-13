import os, re
from typing import TypedDict, List, Optional, Any
from pyswip import Prolog

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END

kb = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kb8.pl")

prolog = Prolog()
if os.path.exists(kb):
    prolog.consult(kb)

docs = [
    Document(page_content="is_a(Animal, Class): dog, cat, bat are mammals; eagle, penguin are birds; salmon, shark are fish."),
    Document(page_content="has_trait(Class, Trait): mammal and bird have warm_blooded; bird has feathers; fish has cold_blooded."),
    Document(page_content="eats(cat, salmon). eats(shark, salmon)."),
    Document(page_content="is_endotherm(Animal) :- has_property(Animal, warm_blooded)."),
    Document(page_content="can_fly(Animal) :- is_a(Animal, bird), Animal \\= penguin. Bats can also fly."),
    Document(page_content="is_carnivore(Animal) :- eats(Animal, Prey), is_a(Prey, _)."),
    Document(page_content="is_predator(Animal) is true if the animal is a carnivore and a mammal."),
    Document(page_content="shares_category(Animal1, Animal2) is true if two different animals belong to the same class."),
    Document(page_content="is_aquatic_predator(Animal) is true if the animal is a fish that eats something."),
    Document(page_content="has_feathers(Animal) is true if the animal has the feathers trait.")
]

db = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = db.as_retriever(search_kwargs={"k": 2})
llm = ChatOpenAI(model="gpt-4o", temperature=0)

grade_prompt = ChatPromptTemplate.from_template(
"""You are a grader assessing whether retrieved context is relevant to a user question.

Context:
{context}

Question:
{question}

Does the context contain facts, traits, or relationships relevant to answering or translating this question?
Answer ONLY 'yes' or 'no'."""
)

translate_prompt = ChatPromptTemplate.from_template(
"""Translate the input question into a single valid Prolog goal syntax based on the context provided.
Use exact predicate names from the context.
Output only the raw Prolog query. No markdown formatting, backticks, or extra text.

Context:
{context}

Question:
{question}"""
)

fix_prompt = ChatPromptTemplate.from_template(
"""Your previous Prolog query threw an error or was invalid.

Question: {question}
Failed Query: {failed_query}
Error: {error}

Context:
{context}

Fix the syntax and output ONLY a valid raw Prolog query call."""
)

class GraphState(TypedDict):
    question: str
    documents: List[Any]
    is_relevant: bool
    retrieval_retry: bool
    query: str
    attempts: List[str]
    retries: int
    error: Optional[str]
    success: bool
    trace: List[str]

def find_preds():
    preds = set()
    for item in prolog.query("current_predicate(Name/Arity)"):
        n = item["Name"]
        if isinstance(n, bytes):
            n = n.decode()
        preds.add((str(n), int(item["Arity"])))
    return preds

KNOWN_PREDS = find_preds()

def split_args(s):
    out = []
    depth = 0
    cur = ""
    for c in s:
        if c == "(":
            depth += 1
            cur += c
        elif c == ")":
            depth -= 1
            cur += c
        elif c == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += c
    if cur.strip():
        out.append(cur.strip())
    return out

def parse_query(q):
    m = re.match(r"^([a-z_][a-zA-Z0-9_]*)\((.*)\)$", q.strip())
    if not m:
        return None, []
    return m.group(1), split_args(m.group(2))

def read_rules(path):
    rules = {}
    if not os.path.exists(path):
        return rules
    with open(path) as f:
        text = f.read()
    text = re.sub(r"%.*", "", text)
    chunks = [c.strip() for c in text.split(".") if c.strip()]
    for c in chunks:
        c = re.sub(r"\s+", " ", c)
        if ":-" not in c:
            m = re.match(r"^([a-z_][a-zA-Z0-9_]*)\((.*?)\)$", c)
            if m:
                p, raw = m.groups()
                args = split_args(raw)
                rules.setdefault((p, len(args)), []).append((args, None))
        else:
            m = re.match(r"^([a-z_][a-zA-Z0-9_]*)\((.*?)\)\s*:\-\s*(.*)$", c)
            if m:
                p, raw, body = m.groups()
                args = split_args(raw)
                rules.setdefault((p, len(args)), []).append((args, body))
    return rules

RULES = read_rules(kb)

def run_prolog(q):
    try:
        res = list(prolog.query(q))
        return bool(res), res, None
    except Exception as e:
        return False, [], str(e)

def apply_vars(expr, mapping):
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[^A-Za-z_]+", expr)
    return "".join(str(mapping.get(t, t)) for t in toks)

def trace(q, env=None):
    env = env or {}
    pred, args = parse_query(q)
    ok, _, _ = run_prolog(q)

    if not pred or (pred, len(args)) not in RULES or not ok:
        return [f"{q} -> {'pass' if ok else 'fail'}"]

    out = [f"{q} -> pass"]
    for head_vars, body in RULES.get((pred, len(args)), []):
        if body is None:
            out.append(f"   matched fact: {q}")
            return out

        vmap = dict(zip(head_vars, args))
        vmap.update(env)

        subgoals = split_args(body)
        steps = []
        valid = True

        for sg in subgoals:
            bound = apply_vars(sg, vmap)
            builtin = any(op in bound for op in ["\\=", "==", " is "])

            if builtin:
                passed, _, _ = run_prolog(bound)
                steps.append(f"   subgoal: {bound} -> {'pass' if passed else 'fail'}")
                if not passed:
                    valid = False
                    break
                continue

            passed, res, _ = run_prolog(bound)
            if not passed:
                steps.append(f"   {bound} -> fail")
                valid = False
                break

            for k, v in res[0].items():
                if isinstance(v, bytes):
                    v = v.decode()
                vmap[k] = str(v)

            next_sg = apply_vars(bound, vmap)
            for line in trace(next_sg, vmap):
                steps.append(f"   {line}")

        if valid:
            out.extend(steps)
            return out

    return [f"{q} -> fail"]

def retrieve_node(state: GraphState):
    q = state["question"]
    matched = retriever.invoke(q)
    return {**state, "documents": matched}

def grade_node(state: GraphState):
    q = state["question"]
    docs = state["documents"]
    context = "\n".join(d.page_content for d in docs)
    res = (grade_prompt | llm | StrOutputParser()).invoke({"context": context, "question": q}).strip().lower()
    return {**state, "is_relevant": "yes" in res}

def expand_retrieval_node(state: GraphState):
    q = state["question"]
    wider = db.as_retriever(search_kwargs={"k": 5})
    matched = wider.invoke(q)
    return {**state, "documents": matched, "retrieval_retry": True}

def translate_node(state: GraphState):
    q = state["question"]
    docs = state["documents"]
    context = "\n".join(d.page_content for d in docs)
    raw = (translate_prompt | llm | StrOutputParser()).invoke({"context": context, "question": q}).strip().rstrip(".")
    return {**state, "query": raw, "attempts": [raw], "retries": 0, "error": None}

def run_node(state: GraphState):
    q = state["query"]
    ok, _, err = run_prolog(q)
    pred, args = parse_query(q)

    if pred and (pred, len(args)) not in KNOWN_PREDS:
        err = f"unknown predicate: {pred}/{len(args)}"

    return {**state, "success": ok, "error": err}

def fix_node(state: GraphState):
    q = state["question"]
    docs = state["documents"]
    context = "\n".join(d.page_content for d in docs)
    fixed = (fix_prompt | llm | StrOutputParser()).invoke({
        "context": context,
        "question": q,
        "failed_query": state["query"],
        "error": state["error"] or "invalid query"
    }).strip().rstrip(".")

    return {
        **state,
        "query": fixed,
        "attempts": state["attempts"] + [fixed],
        "retries": state["retries"] + 1,
        "error": None
    }

def trace_node(state: GraphState):
    q = state["query"]
    err = state["error"]
    ok = state["success"]

    if err:
        out = [f"{q} -> error: {err}"]
    elif ok:
        out = [f"{q} -> TRUE", "deduction trace:"]
        for line in trace(q):
            out.append(f"   {line}")
    else:
        out = [f"{q} -> FALSE"]

    if len(state["attempts"]) > 1:
        out.insert(0, f"retried {len(state['attempts']) - 1} time(s): {state['attempts']}")

    return {**state, "trace": out}

def check_relevancy(state: GraphState):
    if state.get("is_relevant"):
        return "translate"
    if not state.get("retrieval_retry", False):
        return "expand_retrieval"
    return "translate"

def route(state: GraphState):
    if state.get("error") and state.get("retries", 0) < 3:
        return "fix"
    return "trace"

builder = StateGraph(GraphState)

builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade_node)
builder.add_node("expand_retrieval", expand_retrieval_node)
builder.add_node("translate", translate_node)
builder.add_node("run", run_node)
builder.add_node("fix", fix_node)
builder.add_node("trace", trace_node)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", check_relevancy, {"translate": "translate", "expand_retrieval": "expand_retrieval"})
builder.add_edge("expand_retrieval", "translate")
builder.add_edge("translate", "run")
builder.add_conditional_edges("run", route, {"fix": "fix", "trace": "trace"})
builder.add_edge("fix", "run")
builder.add_edge("trace", END)

graph = builder.compile()

if __name__ == "__main__":
    test_questions = [
        "Can an eagle fly?",
        "Can a penguin fly?",
        "Is a dog an endotherm?",
        "Is a bat a predator?",
        "Do a dog and a cat share a category?"
    ]

    for q in test_questions:
        initial = {
            "question": q,
            "documents": [],
            "is_relevant": False,
            "retrieval_retry": False,
            "query": "",
            "attempts": [],
            "retries": 0,
            "error": None,
            "success": False,
            "trace": []
        }

        res = graph.invoke(initial)
        status = "UNKNOWN" if res["error"] else res["success"]

        print(f"Q: {q}")
        print(f"Query: {res['query']}")
        print(f"Status: {status}")
        print("Trace:")
        print("\n".join(res["trace"]))
        print("=" * 40)
