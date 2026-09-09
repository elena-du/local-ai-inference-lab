# 07 - Windows AI APIs

Windows AI APIs are high-level, task-specific WinRT APIs. Microsoft manages system models and hides most runtime details. Official surfaces are C# and C++, not Python, so this lab does not invent a binding. A companion process can expose a narrow local protocol to Python while preserving `GetReadyState()` and `EnsureReadyAsync()` semantics.

Phi Silica is limited access and has a documented 2026 replacement lifecycle; never assume availability. **Exercise:** run `explain windows-ai-api` and design a JSON request/response for a C# bridge that includes readiness evidence.
