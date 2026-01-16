# Project Rules

## Core Principles (Always Apply)

### Autonomous Execution
- Define success criteria before writing any code
- Work in small iterations: write a small piece, run it, verify it works, then continue
- Run code after every change - don't stop after writing, stop after it works
- Regularly check: "Am I still on track to meet the success criteria?"
- Never write large chunks of code without running them

### Simplify Code
- Always write the simplest code that solves the problem
- Prefer straightforward logic over clever tricks
- Use built-in functions instead of custom logic when possible
- After writing, ask: "Can this be shorter? More readable? Are there unnecessary parts?"
- When unsure between two approaches, choose the simpler one

### Config Over Hardcode
- Values that might change belong in config files, not scattered in code
- Config files are the single source of truth for defaults
- No magic numbers - load from config instead
- If a value appears in config, don't duplicate it in code

### No Suboptimal Fallbacks
- When critical operations fail, fail fast with clear error messages
- Don't silently degrade to suboptimal alternatives (no fake data, no useless heuristics)
- Error messages must say: what failed, why it failed, how to fix it
- Let users configure their preferred option rather than automatic fallback chains

## Clean Code Guidelines

### Constants and Naming
- Replace hard-coded values with named constants
- Variables, functions, and classes should reveal their purpose
- Avoid abbreviations unless universally understood

### Comments
- Don't comment what the code does - make code self-documenting
- Use comments to explain WHY something is done a certain way
- Document APIs, complex algorithms, and non-obvious side effects

### Structure
- Each function should do exactly one thing
- Extract repeated code into reusable functions (DRY)
- Keep related code together
- Hide implementation details, expose clear interfaces

### Code Quality
- Refactor continuously
- Fix technical debt early
- Leave code cleaner than you found it

### Workflow
- Make changes file by file
- Don't invent changes beyond what's explicitly requested
- Don't suggest whitespace-only changes
- Verify information before presenting it

## Python Rules (for .py files)

### Style
- Follow Black code formatting (88 char line length)
- Use isort for import sorting
- PEP 8 naming: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- Use absolute imports over relative imports

### Type Hints
- Use type hints for all function parameters and returns
- Use `Optional[Type]` instead of `Type | None`
- Define custom types in `types.py`

### Functions
- Functions should generally be less than 30 lines
- If exceeding 30 lines, extract helper functions or separate concerns

### Testing
- Use pytest for testing
- Write tests for all routes and critical functions
- Use pytest-cov for coverage

### Security
- Use HTTPS in production
- Sanitize all user inputs
- Follow OWASP guidelines

## TypeScript Rules (for .ts/.tsx files)

### Type System
- Prefer interfaces for object definitions
- Use type for unions, intersections, and mapped types
- Avoid `any`, prefer `unknown` for unknown types
- Use strict TypeScript configuration

### Naming
- PascalCase for types and interfaces
- camelCase for variables and functions
- UPPER_CASE for constants
- Prefix React props interfaces with 'Props' (e.g., ButtonProps)

### Functions
- Use explicit return types for public functions
- Prefer async/await over Promises
- Use arrow functions for callbacks

### Best Practices
- Enable strict mode in tsconfig.json
- Use readonly for immutable properties
- Implement proper null checking

## Data Science Rules (for ML/pandas/scikit-learn work)

### Data Handling
- Use pandas for all data operations
- Prefer method chaining for transformations
- Validate data early (schema checks, null handling)
- Use vectorized pandas/numpy operations instead of Python loops

### Feature Engineering
- Create features with modular transformation functions
- Avoid hidden feature engineering inside modeling functions

### Modeling
- Use scikit-learn Pipelines to connect preprocessing and modeling
- Define explicit train() and predict() functions
- Always split data (train/test) properly

### Evaluation
- Always report multiple metrics, not just one
- Build simple evaluation functions that return dicts

### Visualization
- Use plotly for interactive visualizations
- Label axes and add titles in all plots

## Manual Triggers

### Planning (invoke when starting complex work)
- Develop a plan in logical numbered steps (1.1, 1.2, 2.1, etc.)
- Each step should have: goal, tasks, key files, relevant primitives, reference docs, notes
- Task tracking: `[ ]` not started, `[W]` in progress, `[X]` completed, `[S]` skipped
- Read relevant files to understand the problem before planning

### Implementation (invoke when implementing a plan)
- Follow the plan, making changes progressively
- Run tests after changes - never assume they pass
- Update task tracking frequently (after every logical group of edits)
- Do not mark work complete until tests pass
- Check for scope/intent drift compared to original plan

### Critical Reflection (invoke after implementing)
- Check for scope drift or semantic drift
- Check for unintended functionality changes
- Check for assumptions introduced on-the-fly
- Check for duplication (partial or complete)
- Check for incomplete tasks marked as complete

### Work Tracking Verification (invoke to audit progress)
- Verify every task status by inspecting relevant files
- Don't trust the plan - it may be incorrect
- Use `[?]` for uncertain status
