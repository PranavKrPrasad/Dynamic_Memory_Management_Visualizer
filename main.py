from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI(title="Dynamic Memory Management Visualizer")

# =====================================================
# Enums
# =====================================================

class Algorithm(str, Enum):
    FIFO = "FIFO"
    LRU = "LRU"

# =====================================================
# Data Models
# =====================================================

class PageTableEntry(BaseModel):
    """Single entry in a page table."""
    vpn: int
    pfn: Optional[int] = None
    present: bool = False


class ProcessModel(BaseModel):
    """Represents a process."""
    pid: str
    size: int
    page_table: List[PageTableEntry]
    color: Optional[str] = None


class Frame(BaseModel):
    """Represents a physical memory frame."""
    id: int
    pid: Optional[str] = None
    vpn: Optional[int] = None
    is_filled: bool = False
    last_accessed: int = 0     # Used for LRU
    arrival_time: int = 0      # Used for FIFO


class InitConfig(BaseModel):
    frames: int = 8
    algorithm: Algorithm = Algorithm.FIFO


class AccessRequest(BaseModel):
    pid: str
    vpn: int

# =====================================================
# Simulator Core
# =====================================================

class Simulator:
    """Virtual Memory Simulator (FIFO / LRU)."""

    def __init__(self) -> None:
        self.frames: List[Frame] = []
        self.processes: Dict[str, ProcessModel] = {}
        self.frame_capacity: int = 0
        self.algorithm: Algorithm = Algorithm.FIFO
        self.clock: int = 0

        self.total_accesses: int = 0
        self.page_hits: int = 0
        self.page_faults: int = 0

    # ---------------- Initialization ----------------

    def init(self, frames: int, algorithm: Algorithm) -> Dict[str, Any]:
        self.frame_capacity = frames
        self.frames = [Frame(id=i) for i in range(frames)]
        self.processes.clear()
        self.algorithm = algorithm

        self.clock = 0
        self.total_accesses = 0
        self.page_hits = 0
        self.page_faults = 0

        return self.get_state()

    # ---------------- Process Management ----------------

    def create_process(self, pid: str, size: int, color: Optional[str]) -> ProcessModel:
        if pid in self.processes:
            raise ValueError("PID already exists")

        page_table = [PageTableEntry(vpn=i) for i in range(size)]
        process = ProcessModel(
            pid=pid,
            size=size,
            page_table=page_table,
            color=color
        )
        self.processes[pid] = process
        return process

    # ---------------- Memory Access ----------------

    def access(self, pid: str, vpn: int) -> Dict[str, Any]:
        if pid not in self.processes:
            raise ValueError("Unknown PID")

        process = self.processes[pid]
        if vpn < 0 or vpn >= process.size:
            raise ValueError("VPN out of range")

        self.clock += 1
        self.total_accesses += 1

        entry = process.page_table[vpn]
        event = {"time": self.clock, "pid": pid, "vpn": vpn}

        # ---------- Page Hit ----------
        if entry.present and entry.pfn is not None:
            self.page_hits += 1
            frame = self.frames[entry.pfn]
            frame.last_accessed = self.clock

            event.update({
                "result": "hit",
                "pfn": entry.pfn
            })
            return event

        # ---------- Page Fault ----------
        self.page_faults += 1

        # Check for free frame
        free_frame = next((f for f in self.frames if not f.is_filled), None)
        if free_frame:
            self._load_page(pid, vpn, free_frame.id)
            event.update({
                "result": "loaded",
                "pfn": free_frame.id
            })
            return event

        # ---------- Replacement ----------
        victim_frame = self._select_victim()

        if victim_frame.pid is not None and victim_frame.vpn is not None:
            victim_entry = self.processes[victim_frame.pid].page_table[victim_frame.vpn]
            victim_entry.present = False
            victim_entry.pfn = None

        evicted_info = {
            "pid": victim_frame.pid,
            "vpn": victim_frame.vpn
        }

        self._load_page(pid, vpn, victim_frame.id)

        event.update({
            "result": "replaced",
            "pfn": victim_frame.id,
            "evicted": evicted_info
        })
        return event

    # ---------------- Helpers ----------------

    def _load_page(self, pid: str, vpn: int, frame_id: int) -> None:
        frame = self.frames[frame_id]

        frame.pid = pid
        frame.vpn = vpn
        frame.is_filled = True
        frame.last_accessed = self.clock

        if self.algorithm == Algorithm.FIFO:
            frame.arrival_time = self.clock

        entry = self.processes[pid].page_table[vpn]
        entry.present = True
        entry.pfn = frame_id

    def _select_victim(self) -> Frame:
        if self.algorithm == Algorithm.LRU:
            return min(self.frames, key=lambda f: f.last_accessed)
        return min(self.frames, key=lambda f: f.arrival_time)

    # ---------------- State Snapshot ----------------

    def get_state(self) -> Dict[str, Any]:
        return {
            "frames": [f.dict() for f in self.frames],
            "processes": {pid: p.dict() for pid, p in self.processes.items()},
            "metrics": {
                "clock": self.clock,
                "total_accesses": self.total_accesses,
                "page_hits": self.page_hits,
                "page_faults": self.page_faults,
                "hit_ratio": (
                    self.page_hits / self.total_accesses
                    if self.total_accesses > 0 else 0.0
                )
            },
            "algorithm": self.algorithm.value
        }

# =====================================================
# Global Simulator Instance
# =====================================================

memory_simulator = Simulator()

# =====================================================
# Routes
# =====================================================

@app.get("/", response_class=HTMLResponse)
def serve_index() -> str:
    index_path = Path(__file__).parent / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.post("/api/init")
def api_init(config: InitConfig):
    state = memory_simulator.init(config.frames, config.algorithm)
    return JSONResponse({"status": "ok", "state": state})


@app.post("/api/process")
def api_process(payload: Dict[str, Any]):
    try:
        process = memory_simulator.create_process(
            pid=str(payload.get("pid")),
            size=int(payload.get("size", 1)),
            color=payload.get("color")
        )
        return JSONResponse({"status": "ok", "process": process.dict()})
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "error": str(exc)},
            status_code=400
        )


@app.post("/api/access")
def api_access(req: AccessRequest):
    try:
        event = memory_simulator.access(req.pid, req.vpn)
        return JSONResponse({
            "status": "ok",
            "event": event,
            "state": memory_simulator.get_state()
        })
    except Exception as exc:
        return JSONResponse(
            {"status": "error", "error": str(exc)},
            status_code=400
        )


@app.get("/api/state")
def api_state():
    return JSONResponse({
        "status": "ok",
        "state": memory_simulator.get_state()
    })

# =====================================================
# Local Run
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
