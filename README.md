# Game_8puzzle
# 🔢 Giải bài toán 8-Puzzle bằng các thuật toán Tìm kiếm AI

Đây là dự án triển khai bài toán **8-Puzzle** bằng nhiều thuật toán trí tuệ nhân tạo (AI), từ cơ bản đến nâng cao, bao gồm:

- 💡 Tìm kiếm không thông tin (Uninformed Search)
- 🎯 Tìm kiếm có thông tin (Informed Search – Heuristic)
- 🌄 Tìm kiếm cục bộ (Local Search)
- 🤖 Học tăng cường (Reinforcement Learning)

---

## 🧩 Mô tả bài toán

Bài toán **8-Puzzle** là một trò chơi xếp hình trên bảng 3x3 với 8 ô chứa số từ 1 đến 8 và một ô trống. Mục tiêu là đưa các ô về đúng vị trí theo trạng thái đích bằng cách trượt các ô theo 4 hướng: lên, xuống, trái, phải.

Ví dụ trạng thái đích:

1 2 3
4 5 6
7 8 _

yaml
Sao chép
Chỉnh sửa

---

## 🚀 Các thuật toán đã triển khai

### 🔍 Tìm kiếm không thông tin (Uniformed Search)
- BFS (Breadth-First Search)
- DFS (Depth-First Search)
- UCS (Uniform Cost Search)
- DLS, IDDFS

### 🎯 Tìm kiếm có thông tin (Informed Search)
- A* (A star) với heuristic Manhattan hoặc Misplaced Tiles
- Greedy Best-First Search

### 🌄 Tìm kiếm cục bộ (Local Search)
- Hill Climbing (Simple, Steepest Ascent, Stochastic)
- Simulated Annealing
- Local Beam Search
- Genetic Algorithm

### 🤖 Học tăng cường (Reinforcement Learning)
- Q-learning
- SARSA

---

## ⚙️ Yêu cầu cài đặt

Dự án sử dụng Python 3.8+ và các thư viện:
- `pygame` (nếu có giao diện mô phỏng)
- `numpy`, `random`, `matplotlib` (nếu cần hiển thị kết quả)

Cài đặt nhanh:
```bash
pip install -r requirements.txt
▶️ Cách chạy chương trình
Chạy thuật toán bằng dòng lệnh hoặc giao diện:

bash
Sao chép
Chỉnh sửa
python main.py --algorithm astar --heuristic manhattan
Tham số hỗ trợ:

--algorithm: bfs, dfs, ucs, astar, greedy, hill, sarsa, qlearn, ...

--heuristic: manhattan, misplaced

📁 Cấu trúc thư mục
bash
Sao chép
Chỉnh sửa
├── algorithms/
│   ├── uninformed/
│   ├── informed/
│   ├── local_search/
│   ├── reinforcement/
├── gui/               # (nếu có dùng pygame để mô phỏng)
├── utils/
├── main.py
├── README.md
└── requirements.txt
📚 Tài liệu tham khảo
Artificial Intelligence – A Modern Approach (Russell & Norvig)

Lecture Notes & SEED Labs

✍️ Tác giả
Tên: [Tên của bạn]

Liên hệ: [Email, GitHub]
