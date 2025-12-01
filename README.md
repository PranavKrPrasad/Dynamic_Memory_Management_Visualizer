
Virtual Memory Management Simulator
===================================

Contents:
- index.html         : Frontend UI (single-file HTML + JS). Connects to backend at ws://localhost:8000/ws
- main.py            : FastAPI backend implementing simulator, REST endpoints and WebSocket broadcasting
- requirements.txt   : Python requirements
- Dockerfile         : Dockerfile to containerize the backend service
- vm_simulator.zip   : This archive (when created)

Quick start (local):
1. Create and activate a virtualenv (recommended):
   python -m venv venv
   source venv/bin/activate   # on Linux/Mac
   venv\Scripts\activate    # on Windows (PowerShell: .\venv\Scripts\Activate.ps1)

2. Install dependencies & run backend:
   pip install -r requirements.txt
   python main.py

3. Open index.html in your browser (File > Open) OR serve it from a small static server:
   python -m http.server 8080
   then open http://localhost:8080/index.html

Notes:
- The frontend expects the backend at the same machine on port 8000 (ws://localhost:8000/ws).
- If you host frontend from a different origin, add CORS to the FastAPI app (instructions in README).
- For production, run behind a process manager and consider per-session simulators instead of the single global one.
- To run with Docker:
   docker build -t vm-sim .
   docker run -p 8000:8000 vm-sim
