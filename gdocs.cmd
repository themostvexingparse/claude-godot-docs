@echo off
rem Shortcut for the Godot docs search tool: gdocs <command> [args...]
python "%~dp0src\godot_docs.py" %*
exit /b %ERRORLEVEL%
