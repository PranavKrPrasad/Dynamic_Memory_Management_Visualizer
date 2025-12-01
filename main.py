from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import collections
import asyncio
import time
import uvicorn

app = FastAPI(title="VM Simulator API")

class PageTableEntry(BaseModel):
    vpn: int
    pfn: Optional[int] = None
    present: bool = False
    timestamp: int = 0

class ProcessModel(BaseModel):
    pid: str
    size: int
    page_table: List[PageTableEntry]
    color: Optional[str] = None

class Frame(BaseModel):
    id: int
    pid: Optional[str] = None
    vpn: Optional[int] = None
    last_accessed: int = 0
    arrival_time: int = 0
    is_filled: bool = False

class InitConfig(BaseModel):
    frames: int = 8
    algorithm: str = "FIFO"

class AccessRequest(BaseModel):
    pid: str
    vpn: int

class Simulator:
    def __init__(self):
        self.frames: List[Frame] = []
        self.processes: Dict[str, ProcessModel] = {}
        self.frame_capacity = 0
        self.algorithm = "FIFO"
        self.clock = 0
        self.total_accesses = 0
        self.page_hits = 0
        self.page_faults = 0
        self.event_log: List[Dict[str, Any]] = []

    def init(self, frames: int, algorithm: str):
        self.frame_capacity = frames
        self.frames = [Frame(id=i) for i in range(frames)]
        self.algorithm = algorithm
        self.clock = 0
        self.total_accesses = 0
        self.page_hits = 0
        self.page_faults = 0
        self.processes = {}
        self.event_log = []
        return self.state_snapshot()

    def create_process(self, pid: str, size: int, color: Optional[str] = None):
        if pid in self.processes:
            raise ValueError("PID exists")
        page_table = [PageTableEntry(vpn=i) for i in range(size)]
        self.processes[pid] = ProcessModel(pid=pid, size=size, page_table=page_table, color=color)
        self._log_event("process_create", {"pid": pid, "size": size})
        return self.processes[pid]

    def access(self, pid: str, vpn: int):
        if pid not in self.processes:
            raise ValueError("Unknown pid")
        process = self.processes[pid]
        if vpn < 0 or vpn >= process.size:
            raise ValueError("VPN out of range")
        self.clock += 1
        self.total_accesses += 1
        entry = process.page_table[vpn]
        event = {"time": self.clock, "pid": pid, "vpn": vpn}

        if entry.present and entry.pfn is not None:
            self.page_hits += 1
            entry.timestamp = self.clock
            f = self.frames[entry.pfn]
            f.last_accessed = self.clock
            self._log_event("hit", {"pid": pid, "vpn": vpn, "pfn": entry.pfn, "time": self.clock})
            event.update({"result": "hit", "pfn": entry.pfn})
            return event
        else:
            self.page_faults += 1
            self._log_event("fault", {"pid": pid, "vpn": vpn, "time": self.clock})
            free_index = next((i for i, fr in enumerate(self.frames) if not fr.is_filled), None)
            if free_index is not None:
                self._load_into_frame(pid, vpn, free_index)
                self._log_event("load", {"pid": pid, "vpn": vpn, "pfn": free_index})
                event.update({"result": "loaded", "pfn": free_index})
                return event
            else:
                victim = self._select_victim()
                victim_frame = self.frames[victim]
                victim_pid = victim_frame.pid
                victim_vpn = victim_frame.vpn
                if victim_pid and victim_vpn is not None:
                    vp_entry = self.processes[victim_pid].page_table[victim_vpn]
                    vp_entry.present = False
                    vp_entry.pfn = None
                    vp_entry.timestamp = 0
                    self._log_event("evict", {"victim_pid": victim_pid, "victim_vpn": victim_vpn, "pfn": victim})
                self._load_into_frame(pid, vpn, victim)
                self._log_event("load", {"pid": pid, "vpn": vpn, "pfn": victim})
                event.update({"result": "replaced", "pfn": victim, "evicted": {"pid": victim_pid, "vpn": victim_vpn}})
                return event

    def _load_into_frame(self, pid: str, vpn: int, pfn: int):
        frame = self.frames[pfn]
        frame.pid = pid
        frame.vpn = vpn
        frame.is_filled = True
        frame.last_accessed = self.clock
        if self.algorithm.upper() == "FIFO":
            frame.arrival_time = self.clock
        pe = self.processes[pid].page_table[vpn]
        pe.present = True
        pe.pfn = pfn
        pe.timestamp = frame.arrival_time if self.algorithm.upper() == "FIFO" else frame.last_accessed

    def _select_victim(self) -> int:
        if self.algorithm.upper() == "FIFO":
            min_time = float("inf")
            victim = 0
            for f in self.frames:
                if f.arrival_time < min_time:
                    min_time = f.arrival_time
                    victim = f.id
            return victim
        elif self.algorithm.upper() == "LRU":
            min_time = float("inf")
            victim = 0
            for f in self.frames:
                if f.last_accessed < min_time:
                    min_time = f.last_accessed
                    victim = f.id
            return victim
        else:
            return self._select_victim()

    def get_state(self):
        return self.state_snapshot()

    def state_snapshot(self):
        return {
            "frames": [fr.dict() for fr in self.frames],
            "processes": {pid: proc.dict() for pid, proc in self.processes.items()},
            "metrics": {
                "clock": self.clock,
                "total_accesses": self.total_accesses,
                "page_hits": self.page_hits,
                "page_faults": self.page_faults,
                "hit_ratio": (self.page_hits / self.total_accesses) if self.total_accesses > 0 else 0.0
            },
            "algorithm": self.algorithm
        }

    def _log_event(self, etype: str, payload: Dict[str, Any]):
        entry = {"type": etype, "time": time.time(), "sim_clock": self.clock, "payload": payload}
        self.event_log.append(entry)
        if len(self.event_log) > 1000:
            self.event_log.pop(0)

    def get_log(self, limit: int = 100):
        return list(self.event_log[-limit:])

    def run_trace(self, trace: List[Dict[str, Any]], delay: float = 0.0, send_fn=None):
        results = []
        for access in trace:
            try:
                ev = self.access(access["pid"], access["vpn"])
            except Exception as e:
                ev = {"error": str(e), "access": access}
            results.append(ev)
            if send_fn:
                asyncio.create_task(send_fn(ev))
            if delay > 0:
                time.sleep(delay)
        return results

sim = Simulator()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        to_remove = []
        for conn in list(self.active_connections):
            try:
                await conn.send_json(message)
            except Exception:
                to_remove.append(conn)
        for c in to_remove:
            self.disconnect(c)

manager = ConnectionManager()

@app.post("/api/init")
async def api_init(config: InitConfig):
    snapshot = sim.init(frames=config.frames, algorithm=config.algorithm)
    return JSONResponse(content={"status": "ok", "state": snapshot})

@app.post("/api/process")
async def api_create_process(payload: Dict[str, Any]):
    pid = payload.get("pid")
    size = int(payload.get("size", 1))
    color = payload.get("color")
    try:
        proc = sim.create_process(pid, size, color)
        return JSONResponse(content={"status": "ok", "process": proc.dict()})
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=400)

@app.post("/api/access")
async def api_access(access: AccessRequest, background_tasks: BackgroundTasks):
    try:
        ev = sim.access(access.pid, access.vpn)
        background_tasks.add_task(manager.broadcast, {"type": "access_event", "event": ev})
        return JSONResponse(content={"status": "ok", "event": ev, "state": sim.get_state()})
    except Exception as e:
        return JSONResponse(content={"status": "error", "error": str(e)}, status_code=400)

@app.get("/api/state")
async def api_state():
    return JSONResponse(content={"status": "ok", "state": sim.get_state()})

@app.get("/api/metrics")
async def api_metrics():
    return JSONResponse(content={"status": "ok", "metrics": sim.get_state()["metrics"]})

@app.get("/api/logs")
async def api_logs(limit: int = 100):
    return JSONResponse(content={"status": "ok", "logs": sim.get_log(limit)})

@app.post("/api/run_trace")
async def api_run_trace(payload: Dict[str, Any], background_tasks: BackgroundTasks):
    trace = payload.get("trace", [])
    delay = float(payload.get("delay", 0.0))
    async def runner():
        for access in trace:
            try:
                ev = sim.access(access["pid"], access["vpn"])
            except Exception as e:
                ev = {"error": str(e), "access": access}
            await manager.broadcast({"type": "trace_event", "event": ev, "state": sim.get_state()})
            if delay > 0:
                await asyncio.sleep(delay)
    background_tasks.add_task(asyncio.create_task, runner())
    return JSONResponse(content={"status": "ok", "message": "trace started"})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({"type": "init", "state": sim.get_state()})
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)

@app.post("/api/compare")
async def api_compare(payload: Dict[str, Any]):
    trace = payload.get("trace", [])
    frames = int(payload.get("frames", sim.frame_capacity or 8))
    results = {}
    for alg in ("FIFO", "LRU"):
        local = Simulator()
        local.init(frames=frames, algorithm=alg)
        sizes = {}
        for a in trace:
            sizes.setdefault(a["pid"], 0)
            sizes[a["pid"]] = max(sizes[a["pid"]], a["vpn"] + 1)
        for pid, size in sizes.items():
            local.create_process(pid, size=size)
        local.run_trace(trace, delay=0.0)
        results[alg] = local.get_state()["metrics"]
    return JSONResponse(content={"status": "ok", "comparison": results})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
