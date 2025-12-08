from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# -----------------------------------------------------
# FastAPI app
# -----------------------------------------------------

app = FastAPI(title="Dynamic Memory Management Visualizer")


# -----------------------------------------------------
# Data models
# -----------------------------------------------------

class PageTableEntry(BaseModel):
    """Represents an entry in a process's Page Table."""
    vpn: int
    pfn: Optional[int] = None
    present: bool = False
    timestamp: int = 0  # Used for FIFO/LRU


class ProcessModel(BaseModel):
    """Represents a running process."""
    pid: str
    size: int
    page_table: List[PageTableEntry]
    color: Optional[str] = None


class Frame(BaseModel):
    """Represents a physical frame in main memory."""
    id: int
    pid: Optional[str] = None
    vpn: Optional[int] = None
    last_accessed: int = 0
    arrival_time: int = 0
    is_filled: bool = False


class InitConfig(BaseModel):
    frames: int = 8
    algorithm: str = "FIFO"  # FIFO or LRU


class AccessRequest(BaseModel):
    pid: str
    vpn: int


# -----------------------------------------------------
# Simulator core
# -----------------------------------------------------

class Simulator:
    """Virtual memory simulator with FIFO / LRU replacement."""

    def __init__(self) -> None:
        self.frames: List[Frame] = []
        self.processes: Dict[str, ProcessModel] = {}
        self.frame_capacity: int = 0
        self.algorithm: str = "FIFO"
        self.clock: int = 0
        self.total_accesses: int = 0
        self.page_hits: int = 0
        self.page_faults: int = 0

    # ---------- initialization ----------

    def init(self, frames: int, algorithm: str) -> Dict[str, Any]:
        """Initialize or reset the simulation."""
        self.frame_capacity = frames
        self.frames = [Frame(id=i) for i in range(frames)]
        self.processes = {}
        self.algorithm = algorithm.upper()
        self.clock = 0
        self.total_accesses = 0
        self.page_hits = 0
        self.page_faults = 0
        return self._state_snapshot()

    # ---------- process management ----------

    def create_process(self, pid: str, size: int, color: Optional[str]) -> ProcessModel:
        if pid in self.processes:
            raise ValueError("PID already exists")

        page_table = [PageTableEntry(vpn=i) for i in range(size)]
        proc = ProcessModel(pid=pid, size=size, page_table=page_table, color=color)
        self.processes[pid] = proc
        return proc

    # ---------- memory access ----------

    def access(self, pid: str, vpn: int) -> Dict[str, Any]:
        """Simulate accessing a virtual page."""
        if pid not in self.processes:
            raise ValueError("Unknown pid")

        process = self.processes[pid]
        if vpn < 0 or vpn >= process.size:
            raise ValueError("VPN out of range for process")

        self.clock += 1
        self.total_accesses += 1
        entry = process.page_table[vpn]
        event: Dict[str, Any] = {"time": self.clock, "pid": pid, "vpn": vpn}

        # --- Hit ---
        if entry.present and entry.pfn is not None:
            self.page_hits += 1
            pfn = entry.pfn
            entry.timestamp = self.clock
            self.frames[pfn].last_accessed = self.clock
            event.update({"result": "hit", "pfn": pfn})
            return event

        # --- Page fault ---
        self.page_faults += 1

        # free frame?
        free_index = next((i for i, fr in enumerate(self.frames) if not fr.is_filled), None)
        if free_index is not None:
            self._load_into_frame(pid, vpn, free_index)
            event.update({"result": "loaded", "pfn": free_index})
            return event

        # replacement needed
        victim_index = self._select_victim()
        victim_frame = self.frames[victim_index]

        # invalidate victim page
        if victim_frame.pid is not None and victim_frame.vpn is not None:
            victim_entry = self.processes[victim_frame.pid].page_table[victim_frame.vpn]
            victim_entry.present = False
            victim_entry.pfn = None
            victim_entry.timestamp = 0

        self._load_into_frame(pid, vpn, victim_index)
        event.update(
            {
                "result": "replaced",
                "pfn": victim_index,
                "evicted": {"pid": victim_frame.pid, "vpn": victim_frame.vpn},
            }
        )
        return event

    # ---------- helpers ----------

    def _load_into_frame(self, pid: str, vpn: int, pfn: int) -> None:
        frame = self.frames[pfn]
        frame.pid = pid
        frame.vpn = vpn
        frame.is_filled = True
        frame.last_accessed = self.clock
        if self.algorithm == "FIFO":
            frame.arrival_time = self.clock

        entry = self.processes[pid].page_table[vpn]
        entry.present = True
        entry.pfn = pfn
        entry.timestamp = frame.arrival_time if self.algorithm == "FIFO" else frame.last_accessed

    def _select_victim(self) -> int:
        if self.algorithm == "LRU":
            key = lambda f: f.last_accessed
        else:
            key = lambda f: f.arrival_time

        victim = min(self.frames, key=key)
        return victim.id

    def _state_snapshot(self) -> Dict[str, Any]:
        return {
            "frames": [fr.dict() for fr in self.frames],
            "processes": {pid: proc.dict() for pid, proc in self.processes.items()},
            "metrics": {
                "clock": self.clock,
                "total_accesses": self.total_accesses,
                "page_hits": self.page_hits,
                "page_faults": self.page_faults,
                "hit_ratio": (self.page_hits / self.total_accesses)
                if self.total_accesses > 0
                else 0.0,
            },
            "algorithm": self.algorithm,
        }


# single global simulator
sim = Simulator()


# -----------------------------------------------------
# Routes
# -----------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    """Serve the main HTML page."""
    index_path = Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.post("/api/init")
def api_init(config: InitConfig):
    snapshot = sim.init(frames=config.frames, algorithm=config.algorithm)
    return JSONResponse({"status": "ok", "state": snapshot})


@app.post("/api/process")
def api_process(payload: Dict[str, Any]):
    pid = str(payload.get("pid"))
    size = int(payload.get("size", 1))
    color = payload.get("color")

    try:
        proc = sim.create_process(pid, size, color)
        return JSONResponse({"status": "ok", "process": proc.dict()})
    except Exception as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)


@app.post("/api/access")
def api_access(req: AccessRequest):
    try:
        ev = sim.access(req.pid, req.vpn)
        return JSONResponse({"status": "ok", "event": ev, "state": sim._state_snapshot()})
    except Exception as exc:
        return JSONResponse({"status": "error", "error": str(exc)}, status_code=400)


@app.get("/api/state")
def api_state():
    return JSONResponse({"status": "ok", "state": sim._state_snapshot()})


# -----------------------------------------------------
# Local run
# -----------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

