# ShopConnect Object-Centric Intelligence Environment (Phase 1)

A controlled, deterministic Python research testbed designed to investigate whether **persistent domain objects with progressive signature disclosure** outperform conventional LLM interaction paradigms (monolithic full-state regeneration and flat tool-calling agents) in stateful, multi-step meal-planning workflows.

**Project Status:** Active Research Prototype (Stage 4 Complete, Stage 5 Pending)
**Verified Local Test Suite:** 86 Passed / 0 Failed (Python 3.14.7, pytest 9.2.0)
**Documentation:** [Master Research Specification](docs/RESEARCH_SPEC.md) | [Implementation Status Report](docs/IMPLEMENTATION_STATUS.md)

---

## 1. Research Motivation & Core Questions

### The Challenge of Stateful LLM Workflows
In stateful applications—such as weekly meal planners, itinerary builders, or configuration managers—interaction designs typically rely on one of two paradigms:
1. **Full-State Regeneration (Condition A):** The entire application state (e.g., all 7 days, 21 meals, recipe ingredients, and the grocery basket) is serialized into the model prompt on every turn. The LLM is asked to output the complete updated state. This approach incurs high token costs, increases latency, and risks **unintended state drift** (hallucinated mutations to unrelated, non-target slots).
2. **Flat Tool-Calling (Condition B):** The agent receives flat API function definitions alongside monolithic state snapshots, without object encapsulation or progressive disclosure.

### The Object-Centric Hypothesis (Condition C)
ShopConnect investigates an alternative architecture: treating Python domain objects as **persistent in-memory sources of truth**. 
* The LLM is initially exposed only to **compact class signatures, read-only inspection accessors, and allowlisted action proposals** (e.g., proposing `replace_meal` executed via `ActionBroker`).
* Fine-grained details (such as ingredient requirements for an individual recipe) are inspected **selectively on demand**.
* The LLM emits a small, strongly typed action proposal. Trusted application code validates domain invariants, stages the change on an isolated candidate copy, recomputes derived state, and commits atomically.

### Research Questions
* **Primary (RQ1):** Can a persistent object-centric runtime with selective context and typed operations reduce prompt token consumption without degrading state correctness, constraint preservation, or task completion rates?
* **Cost & Token Efficiency (RQ2):** Does selective signature disclosure decrease input prompt tokens and cumulative cost compared to full-plan regeneration and flat tool-calling?
* **State Preservation (RQ3):** Does encapsulated domain method invocation reduce or eliminate unintended mutations in unmentioned meal slots relative to full-plan regeneration?
* **Failure Dynamics (RQ4):** Can a deterministic validation gate intercept invalid or constraint-violating proposals with zero committed state violations?

> [!NOTE]
> These research questions represent empirical evaluation hypotheses to be measured across standardized benchmark tasks in Stage 6, rather than pre-assumed results.

---

## 2. Current Implementation (Stages 1, 2, 3 & 4)

The project has completed foundational domain modeling, deterministic operations, the complete validation/transaction safety layer, and comparative context builders. All implemented modules run offline with zero external API or LLM dependencies.

### Stage 1: Domain Entities & Seed Catalog
* **Core Models ([`src/shopconnect_env/domain/models.py`](src/shopconnect_env/domain/models.py)):** Implemented pure Python `@dataclass` structures:
  * `Ingredient`: Immutable frozen dataclass with canonical ID, display name, category, standard unit, and allergen classifications.
  * `IngredientRequirement`: Quantity (`Decimal`) and unit specification with a pure `scale(factor)` method.
  * `Recipe`: Recipe blueprint with portion sizes, tags, `estimated_cost` (`Decimal`), and `cost_per_serving`.
  * `Meal`: Scheduled meal slot (`day`, `meal_type`, `recipe_id`, `servings`, `slot_key`, `calculate_cost`).
  * `UserPreferences`: Household constraints (`diet_tags`, `allergies`, `weekly_budget`, `household_size`).
  * `MealPlan`: Authoritative 7-day schedule (21 slots) with monotonic `plan_version` tracking.
  * `GroceryItem` & `GroceryBasket`: Immutable derived projections representing aggregated shopping requirements.
* **Synthetic Catalog ([`data/catalog_seed.json`](data/catalog_seed.json)):** Curated dataset containing 31 canonical ingredients with allergen metadata and 18 realistic recipes with 100% referential integrity verified at parse time.

### Stage 2: Deterministic Operations & Derived Basket Math
* **State Mutations ([`src/shopconnect_env/operations/mutations.py`](src/shopconnect_env/operations/mutations.py)):**
  * `apply_replace_meal(...)`: Replaces a targeted slot on an isolated `copy.deepcopy()` candidate, verifies recipe existence, validates positive integer servings, leaves all 20 other slots byte-identical, and increments `plan_version`.
  * `apply_change_servings(...)`: Updates portion sizes on a candidate clone without mutating the original plan.
* **Derived Basket Engine ([`src/shopconnect_env/operations/basket_calculator.py`](src/shopconnect_env/operations/basket_calculator.py)):**
  * `calculate_grocery_basket(...)`: Deterministically inspects active meals, computes exact scaled requirements via $\text{scale\_factor} = \frac{\text{meal.servings}}{\text{recipe.default\_servings}}$, aggregates shared ingredients across recipes using `Decimal` math, verifies unit consistency, and returns an immutable `GroceryBasket`.

### Stage 3: Validation, Policy Engine & Action Broker
* **Pydantic Action Schemas ([`src/shopconnect_env/validation/schemas.py`](src/shopconnect_env/validation/schemas.py)):** Strict Pydantic v2 schemas (`ReplaceMealArgs`, `ChangeServingsArgs`, `InspectObjectArgs`, `ActionProposal`, `ActionResult`) rejecting unexpected fields (`extra="forbid"`), validating coordinates, and explicitly disallowing float or boolean servings.
* **Policy Engine ([`src/shopconnect_env/validation/policy_engine.py`](src/shopconnect_env/validation/policy_engine.py)):** Deterministic validator enforcing dietary compliance, allergen safety with conservative unknown-metadata rejection, weekly budget limits, and non-target slot invariants.
* **Action Broker ([`src/shopconnect_env/validation/broker.py`](src/shopconnect_env/validation/broker.py)):** Transactional coordinator managing isolated candidate staging, stale-version checks, policy enforcement, derived basket recalculation, and atomic state replacement (`self._state = RuntimeState(...)`).

### Stage 4: Comparative Context Strategies (Conditions A, B & C)
* **Shared Context Protocol ([`src/shopconnect_env/context/base.py`](src/shopconnect_env/context/base.py)):** Standardized `ContextRequest` and `ContextResult` data models; deterministic Decimal-safe serializers (`serialize_all_21_slots`, `serialize_recipe`, `serialize_ingredient`, `serialize_grocery_basket`).
* **Condition A: Full-State Regeneration ([`src/shopconnect_env/context/full_state.py`](src/shopconnect_env/context/full_state.py)):** Monolithic baseline serializing all 21 slots (with explicit unfilled slot representations), full recipe ingredient requirements, complete recipe catalog, raw ingredient catalog, and grocery basket.
* **Condition B: Conventional Tool Agent ([`src/shopconnect_env/context/tool_context.py`](src/shopconnect_env/context/tool_context.py)):** Flat tool-calling baseline exposing function specifications for `replace_meal` and `change_servings` directly derived from Stage 3 Pydantic schemas, accompanied by a clean schedule and recipe summary snapshot.
* **Condition C: Object-Centric Progressive Disclosure ([`src/shopconnect_env/context/object_context.py`](src/shopconnect_env/context/object_context.py)):** Compact class signatures (`MealPlan`, `Recipe`, `Ingredient`, `UserPreferences`), user constraints, and optional target slot summary, paired with on-demand read-only inspection methods (`inspect_slot`, `inspect_recipe`, `inspect_recipe_ingredients`, `inspect_ingredient`, `inspect_plan`, `inspect_preferences`), clearly separated allowlisted ActionBroker proposals (`replace_meal`, `change_servings`), and complete disclosure audit logging.

### Verified Local Test Results
The test suite currently contains **86 unit tests** passing cleanly:
* `tests/test_domain_models.py` (9 tests)
* `tests/test_operations.py` (10 tests)
* `tests/test_validation_schemas.py` (12 tests)
* `tests/test_policy_engine.py` (14 tests)
* `tests/test_action_broker.py` (15 tests)
* `tests/test_context_builders.py` (26 tests)

---

## 3. Proposed Target Architecture (Stages 4–6)

The full ShopConnect research runtime maintains a strict **Trust Boundary** between the untrusted agent and authoritative application state.

```
+---------------------------------------------------------------------------------------+
|                                    TRUST BOUNDARY                                     |
|                                                                                       |
|   +-----------------------+              ActionProposal             +-------------+   |
|   |   LLM Agent / Model   | --------------------------------------> |  Pydantic   |   |
|   |  (Untrusted Proposer) | <-------------------------------------- |   Schema    |   |
|   +-----------------------+           ActionResult / Reject         +------+------+   |
|                                                                            |          |
| ===========================================================================|========= |
|                                                                            v          |
|   +-------------------------------------------------------------------------------+   |
|   |                Deterministic Application Core (Implemented in Stage 3)        |   |
|   |                                                                               |   |
|   |   +-------------------+    +---------------------+    +-------------------+   |   |
|   |   |   Policy Engine   | -> | Candidate Staging   | -> | Transaction Commit|   |   |
|   |   |  (Preconditions)  |    | & Derived Recalc    |    | & Postconditions  |   |   |
|   |   +-------------------+    +---------------------+    +-------------------+   |   |
|   |                                                                               |   |
|   |   +-----------------------------------------------------------------------+   |   |
|   |   |                Persistent State Store (Implemented in Stages 1-2)     |   |   |
|   |   |  [MealPlan v(N)]  <-->  [Meals / Recipes]  --->  [Derived Grocery]   |   |   |
|   |   +-----------------------------------------------------------------------+   |   |
|   +-------------------------------------------------------------------------------+   |
+---------------------------------------------------------------------------------------+
```

### Planned Target Request Lifecycle
1. **Context Construction (Stage 4):** A `ContextBuilder` formats the prompt under Condition A (full state), Condition B (flat tools), or Condition C (progressive signature disclosure).
2. **Action Proposal (Stage 5):** The model emits a structured JSON action payload.
3. **Pydantic Validation (Implemented in Stage 3):** Untrusted JSON is parsed against strict typed schemas.
4. **Policy Precondition Check (Implemented in Stage 3):** The `PolicyEngine` checks dietary tags and budget ceilings.
5. **Candidate Staging (Implemented in Stage 3):** The `ActionBroker` stages the change on a cloned `MealPlan`.
6. **Deterministic Execution (Implemented in Stage 2):** Allowlisted functions (`apply_replace_meal`) mutate the candidate.
7. **Pure Recalculation (Implemented in Stage 2):** `calculate_grocery_basket` re-derives the shopping list.
8. **Invariant Verification (Implemented in Stage 3):** Postcondition checks assert that non-target slots were not mutated.
9. **Atomic Commit or Rollback (Implemented in Stage 3):** Valid proposals commit with an incremented version; failures revert with zero state mutation.

---

## 4. Repository Structure

Below is the repository layout distinguishing **existing implemented files**, **empty directories reserved for upcoming stages**, and **planned modules**:

```text
shopconnect-object-env/
├── .gitignore                         # Python cache, virtual environment, and OS ignores
├── pytest.ini                         # Pytest path and runner configuration
├── requirements.txt                   # Pinned direct development dependencies
├── README.md                          # Project overview and reproduction guide (this file)
│
├── data/
│   └── catalog_seed.json              # [Implemented] 31 ingredients and 18 recipes with allergens
│
├── docs/
│   ├── RESEARCH_SPEC.md               # [Implemented] Master architectural blueprint
│   └── IMPLEMENTATION_STATUS.md       # [Implemented] Technical verification and milestone report
│
├── src/
│   └── shopconnect_env/               # Core Python package
│       ├── __init__.py                # [Implemented] Package version metadata
│       │
│       ├── domain/                    # [Stage 1 - Implemented]
│       │   ├── __init__.py            # Domain exports
│       │   └── models.py              # Ingredient, Recipe, Meal, MealPlan, GroceryBasket
│       │
│       ├── operations/                # [Stage 2 - Implemented]
│       │   ├── __init__.py            # Operations exports
│       │   ├── mutations.py           # apply_replace_meal, apply_change_servings
│       │   └── basket_calculator.py   # calculate_grocery_basket
│       │
│       ├── validation/                # [Stage 3 - Implemented]
│       │   ├── __init__.py            # Validation exports
│       │   ├── schemas.py             # ActionProposal, ReplaceMealArgs, ActionResult
│       │   ├── policy_engine.py       # Dietary, allergen, budget, invariant verifier
│       │   └── broker.py              # ActionBroker transaction coordinator
│       │
│       ├── context/                   # [Stage 4 - Directory Present, Modules Planned]
│       │   └── (planned: full_state.py, tool_context.py, object_context.py)
│       │
│       ├── agents/                    # [Stage 5 - Directory Present, Modules Planned]
│       │   └── (planned: base.py, mock_agent.py, llm_adapter.py)
│       │
│       └── evaluation/                # [Stage 6 - Directory Present, Modules Planned]
│           └── (planned: metrics.py, runner.py)
│
├── tests/
│   ├── test_domain_models.py          # [Stage 1 - Implemented] 9 passing unit tests
│   ├── test_operations.py             # [Stage 2 - Implemented] 10 passing unit tests
│   ├── test_validation_schemas.py     # [Stage 3 - Implemented] 12 passing unit tests
│   ├── test_policy_engine.py          # [Stage 3 - Implemented] 11 passing unit tests
│   └── test_action_broker.py          # [Stage 3 - Implemented] 11 passing unit tests
│
└── results/                           # [Directory Present] Reserved for benchmark execution traces
```

---

## 5. Portable Environment Setup & Installation

These instructions apply to any developer cloning the repository on a Windows workstation.

### Prerequisites
* **Python 3.14 (or Python 3.11+):** Must be installed on your machine and available in your system path.

### 1. Clone the Repository
Open PowerShell and clone the repository:
```powershell
git clone https://github.com/Suni1174/shopconnect-object-env.git
cd shopconnect-object-env
```

### 2. Create a Virtual Environment (if not already present)
```powershell
python -m venv .venv
```

### 3. Install Direct Dependencies
Install the pinned development dependencies directly using the virtual environment's Python interpreter:
```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Activate Environment (Optional Convenience)
Activating the virtual environment is an optional convenience that allows you to run `pytest` or `python` without prefixing the path:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell blocks script execution, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

---

## 6. Running the Automated Tests

The test suite can be run at any time using the direct path to the virtual environment's Python executable:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

### Latest Verified Local Test Run
```text
============================= test session starts =============================
platform win32 -- Python 3.14.7, pytest-9.2.0, pluggy-1.7.0 -- .venv\Scripts\python.exe
rootdir: <cloned-repo-path>
configfile: pytest.ini
testpaths: tests
collecting ... collected 85 items

tests/test_action_broker.py::test_successful_meal_replacement PASSED     [  1%]
...
tests/test_context_builders.py::test_condition_c_cumulative_character_tracking PASSED [ 47%]
...
tests/test_domain_models.py::test_derived_grocery_basket_calculation PASSED [ 57%]
...
tests/test_operations.py::test_recalculate_basket_after_replacement PASSED [ 69%]
...
tests/test_policy_engine.py::test_invariant_validation_detects_unintended_changes PASSED [ 82%]
...
tests/test_validation_schemas.py::test_inspect_object_args_validation PASSED [100%]

============================= 85 passed in 0.61s ==============================
```

---

## 7. Implementation Roadmap & Milestones

| Stage | Milestone | Status | Key Modules & Deliverables |
| :--- | :--- | :---: | :--- |
| **Stage 1** | Core Domain Models & Seed Data | **Completed** | `domain/models.py`, `catalog_seed.json` (9 passing tests). |
| **Stage 2** | Deterministic Operations & Basket Math | **Completed** | `operations/mutations.py`, `operations/basket_calculator.py` (10 passing tests). |
| **Stage 3** | Validation, PolicyEngine & ActionBroker | **Completed** | Pydantic action schemas, candidate staging, and atomic commit/rollback (41 passing tests). |
| **Stage 4** | Comparative Context Strategies (A, B, C) | **Completed** | `context/base.py`, `context/full_state.py`, `context/tool_context.py`, `context/object_context.py` (25 passing tests). |
| **Stage 5** | Mock Agent & Offline Test Pipeline | *Pending* | Deterministic mock agent, zero-cost offline test harness. |
| **Stage 6** | Real LLM Adapter, Tasks & Benchmark Runner | *Pending* | Live model adapter, 25 frozen tasks, `results/comparison.csv`. |

For additional architectural details and technical references:
* [Master Research Specification](docs/RESEARCH_SPEC.md)
* [Implementation Status Report](docs/IMPLEMENTATION_STATUS.md)
