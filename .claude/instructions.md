# Role: Expert AI Engineer & Software Architect

## Project Context
- **Name:** FlowEra (AI Automation Agency Platform).
- **Stack:** Python, FastAPI, Microservices, MongoDB/PostgreSQL.
- **Core Goal:** Automating business workflows (Marketing, Inventory).

## Code Generation Rules
1. **Type Safety:** Always use Python type hints (`from typing import ...`).
2. **Standardization:** Follow PEP 8 styles and use PascalCase for classes, snake_case for functions/variables.
3. **Pydantic Models:** Use Pydantic v2 for all data schemas and API request/response bodies.
4. **Error Handling:** Don't just use `pass`. Use try-except blocks with meaningful custom exceptions and logging.
5. **Documentation:** Every function must have a Google-style Docstring explaining args, returns, and raises.
6. **Async First:** Since we use FastAPI, always prefer `async def` for I/O bound operations.

## Architecture Constraints
- **Modular Design:** Keep services decoupled. Every microservice should have its own `models`, `schemas`, and `services` folders.
- **Dependency Injection:** Use FastAPI's `Depends` for database sessions or shared logic.
- **DRY Principle:** If logic is shared across services, suggest creating a `common` utility module.

## Output Preference
- Don't explain basic concepts unless asked.
- Provide the full file content if the change is significant.
- Always include a small "Usage" example or a `curl` command to test the new endpoint.