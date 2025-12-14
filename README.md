# 🧠 Dynamic Memory Management Visualizer

**Virtual Memory & Paging Simulator (FIFO / LRU)**

An interactive web-based simulator to visualize **virtual memory management**, **page tables**, and **page replacement algorithms** using a modern UI and a FastAPI backend.

---

## 📌 Features

* Virtual → Physical address translation
* Page Table visualization
* Physical memory frame allocation
* Page replacement algorithms:

  * FIFO (First-In First-Out)
  * LRU (Least Recently Used)
* Performance metrics:

  * Page hits
  * Page faults
  * Hit ratio
* Clean, dark-themed interactive UI

---

## 🧩 Tech Stack

* **Frontend**: HTML, CSS (Tailwind), JavaScript
* **Backend**: Python, FastAPI
* **Server**: Uvicorn
* **Visualization**: DOM-based dynamic rendering

---

## 📁 Project Structure

```
Dynamic_Memory_Management_Visualizer/
│
├── index.html          # Frontend UI
├── main.py             # FastAPI backend
├── requirements.txt    # Python dependencies
├── Dockerfile          # (Optional) Docker support
├── README.md
└── .venv/              # Python virtual environment
```

---

## ⚠️ IMPORTANT (Read This First)

❌ **Do NOT open `index.html` directly**
❌ **Do NOT use VS Code Live Server**

✅ This project **must be run via the FastAPI backend**, otherwise fetch errors will occur.

---

## ✅ Step-by-Step: How to Run the Project (Recommended)

### 1️⃣ Open Terminal in Project Folder

```bash
cd Dynamic_Memory_Management_Visualizer
```

---

### 2️⃣ (Optional but Recommended) Activate Virtual Environment

#### Windows (PowerShell):

```bash
.venv\Scripts\activate
```

---

### 3️⃣ Install Required Dependencies

```bash
python -m pip install -r requirements.txt
```

> If you see **“Requirement already satisfied”**, that is **normal**.

---

### 4️⃣ Start the Backend Server

```bash
python -m uvicorn main:app
```

✅ You should see:

```
Uvicorn running on http://127.0.0.1:8000
Application startup complete.
```

---

### 5️⃣ Open the Application in Browser

Open **ONLY** this URL:

```
http://127.0.0.1:8000
```

🚫 Do **NOT** open `index.html`
🚫 Do **NOT** use port `5500`

---

## 🧪 Backend Verification (Optional)

To verify backend is running correctly, open:

```
http://127.0.0.1:8000/docs
```

This opens FastAPI’s Swagger UI.

---

## 🛠 Common Issues & Solutions

### ❌ “Failed to fetch” / “Unexpected end of JSON input”

**Cause:** Frontend opened without backend
**Solution:**
✔ Start backend first
✔ Open app via `http://127.0.0.1:8000`

---

### ❌ `uvicorn` not recognized

Use:

```bash
python -m uvicorn main:app
```

(Recommended for Windows)

---

### ❌ Server restarting continuously

Run without reload:

```bash
python -m uvicorn main:app
```

---

## 🎓 Academic Use

This project is suitable for:

* Operating Systems coursework
* Memory management demonstrations
* Paging & page replacement algorithm visualization
* Mini-project / lab evaluation

---

## 🚀 Future Enhancements

* Add segmentation & virtual memory swapping
* Add more algorithms (Optimal, Clock)
* Graph-based memory access timeline
* Export simulation results

---
