# LangGraph Logical Inference Engine

An AI engine combining **LangChain/GPT-4o** and **PySwip (Prolog)** using **LangGraph**. It converts natural language into Prolog queries, executes them against a knowledge base (`kb8.pl`), and then generates reasoning traces.

## 🚀 Features
- LangGraph Flow: Built as a state machine to handle loops, routing, and state transitions.
- Context Grading: Checks if RAG results are relevant before translating, increasing retrieval size if needed.
- Self-Refining Queries: Catches Prolog syntax and runtime errors and retries failed queries automatically.
- Logic Tracing: Prints the full step-by-step resolution path for every query execution.
## 🛠 Setup
1. **Install SWI-Prolog** (Required for PySwip):
   * macOS: `brew install swi-prolog`
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
3. **Set Your API Key**:
   ```bash
    export OPENAI_API_KEY="your-api-key
3. **RUN**:
   ```bash
   python 3 main.py
