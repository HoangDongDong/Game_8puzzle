# Game_8puzzle
# 🔢 Giải bài toán 8-Puzzle bằng các thuật toán Tìm kiếm AI

Đây là dự án triển khai bài toán **8-Puzzle** bằng nhiều thuật toán trí tuệ nhân tạo (AI), từ cơ bản đến nâng cao, bao gồm:

- 💡 Tìm kiếm không thông tin (Uninformed Search)
- 🎯 Tìm kiếm có thông tin (Informed Search – Heuristic)
- 🌄 Tìm kiếm cục bộ (Local Search)
- 🤖 Học tăng cường (Reinforcement Learning)
- 🎯Cpps


## 🧩 Mô tả bài toán

Bài toán **8-Puzzle** là một trò chơi xếp hình trên bảng 3x3 với 8 ô chứa số từ 1 đến 8 và một ô trống. Mục tiêu là đưa các ô về đúng vị trí theo trạng thái đích bằng cách trượt các ô theo 4 hướng: lên, xuống, trái, phải.

Ví dụ trạng thái đích:

1 2 3
4 5 6
7 8 0

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
- `numpy`, `random`, `matplotlib`.... (nếu cần hiển thị kết quả)

▶️ Cách chạy chương trình
Chạy thuật toán bằng dòng lệnh hoặc giao diện:

bash
Sao chép
Chỉnh sửa
python main.py --algorithm astar --heuristic manhattan
Tham số hỗ trợ:

--algorithm: bfs, dfs, ucs, astar, greedy, hill, sarsa, qlearn, ...

--heuristic: manhattan, misplaced


✍️ Tác giả
Tên: [Hoàng Văn Đông]
