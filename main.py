from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import math

app = FastAPI()

# ─────────────────────────────────────────
# AI 로직 (Minimax + Alpha-Beta Pruning)
# ─────────────────────────────────────────

SIZE = 15
WIN = 5
INF = 100_000_000

DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

# (연속 개수, 열린 끝 수) → 점수
PATTERN = {
    (5, 0): INF, (5, 1): INF, (5, 2): INF,
    (4, 2): 2_000_000,   # 활사 — 다음 수 승리
    (4, 1): 200_000,     # 반활사
    (4, 0): 50_000,
    (3, 2): 80_000,      # 활삼
    (3, 1): 8_000,
    (3, 0): 500,
    (2, 2): 2_000,
    (2, 1): 400,
    (2, 0): 20,
    (1, 2): 50,
    (1, 1): 5,
    (1, 0): 0,
}

SEARCH_DEPTH = 5
MAX_CANDIDATES = 16


def check_win(board, player):
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != player:
                continue
            for dr, dc in DIRECTIONS:
                count = 1
                nr, nc = r + dr, c + dc
                while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == player:
                    count += 1
                    nr += dr
                    nc += dc
                if count >= WIN:
                    return True
    return False


def score_segment(board, r, c, player, opp, dr, dc):
    """한 방향으로 r,c부터 시작하는 연속 돌의 패턴 점수"""
    count = 0
    nr, nc = r, c
    while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == player:
        count += 1
        nr += dr
        nc += dc

    if count == 0:
        return 0

    # 앞쪽 열린 끝 확인 (시작점 앞)
    pr, pc = r - dr, c - dc
    front_open = 0 <= pr < SIZE and 0 <= pc < SIZE and board[pr][pc] == 0

    # 뒤쪽 열린 끝 확인 (연속 끝 다음)
    back_open = 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == 0

    open_ends = (1 if front_open else 0) + (1 if back_open else 0)
    key = (min(count, 5), open_ends)
    return PATTERN.get(key, 0)


def evaluate(board, ai, human):
    ai_score = 0
    human_score = 0
    visited_ai = set()
    visited_human = set()

    for r in range(SIZE):
        for c in range(SIZE):
            cell = board[r][c]
            if cell == 0:
                continue
            for dr, dc in DIRECTIONS:
                # 이미 이 방향 세그먼트를 앞쪽 돌이 처리했으면 스킵
                pr, pc = r - dr, c - dc
                if 0 <= pr < SIZE and 0 <= pc < SIZE and board[pr][pc] == cell:
                    continue
                key = (r, c, dr, dc)
                if cell == ai and key not in visited_ai:
                    visited_ai.add(key)
                    ai_score += score_segment(board, r, c, ai, human, dr, dc)
                elif cell == human and key not in visited_human:
                    visited_human.add(key)
                    human_score += score_segment(board, r, c, human, ai, dr, dc)

    return ai_score - int(human_score * 1.25)


def winning_moves(board, player):
    """한 수 두면 바로 5목이 되는 칸"""
    moves = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != 0:
                continue
            board[r][c] = player
            if check_win(board, player):
                moves.append((r, c))
            board[r][c] = 0
    return moves


def open_four_moves(board, player):
    """활사(열린 4)를 만드는 수 — 상대는 반드시 막아야 함"""
    opp = 3 - player
    moves = []
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] != 0:
                continue
            board[r][c] = player
            if score_segment_all(board, r, c, player, opp) >= 2_000_000:
                moves.append((r, c))
            board[r][c] = 0
    return moves


def pick_best_move(board, moves, player):
    """여러 필수 수 중 평가가 가장 좋은 수 선택"""
    if len(moves) == 1:
        return moves[0]
    opp = 3 - player
    best, best_score = moves[0], -math.inf
    for r, c in moves:
        board[r][c] = player
        score = evaluate(board, player, opp)
        board[r][c] = 0
        if score > best_score:
            best_score = score
            best = (r, c)
    return best


def get_candidates(board):
    """기존 돌 주변 2칸 후보 + 위협 우선 정렬"""
    visited = set()
    for r in range(SIZE):
        for c in range(SIZE):
            if board[r][c] == 0:
                continue
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == 0:
                        visited.add((nr, nc))

    if not visited:
        return [(SIZE // 2, SIZE // 2)]

    # 위협 점수로 정렬 (높은 게 먼저)
    def threat(pos):
        r, c = pos
        board[r][c] = 2
        s2 = score_segment_all(board, r, c, 2, 1)
        board[r][c] = 1
        s1 = score_segment_all(board, r, c, 1, 2)
        board[r][c] = 0
        return s2 + s1

    ordered = sorted(visited, key=threat, reverse=True)
    return ordered[:MAX_CANDIDATES]


def score_segment_all(board, r, c, player, opp):
    total = 0
    for dr, dc in DIRECTIONS:
        # 세그먼트 시작점 찾기 (역방향으로 같은 돌 따라가기)
        sr, sc = r, c
        while 0 <= sr - dr < SIZE and 0 <= sc - dc < SIZE and board[sr - dr][sc - dc] == player:
            sr -= dr
            sc -= dc
        total += score_segment(board, sr, sc, player, opp, dr, dc)
    return total


def minimax(board, depth, alpha, beta, is_max, ai, human):
    if check_win(board, ai):
        return INF + depth * 1000, None
    if check_win(board, human):
        return -(INF + depth * 1000), None

    candidates = get_candidates(board)
    if depth == 0 or not candidates:
        return evaluate(board, ai, human), None

    best_move = candidates[0]
    if is_max:
        best_val = -math.inf
        for r, c in candidates:
            board[r][c] = ai
            val, _ = minimax(board, depth - 1, alpha, beta, False, ai, human)
            board[r][c] = 0
            if val > best_val:
                best_val = val
                best_move = (r, c)
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_move
    else:
        best_val = math.inf
        for r, c in candidates:
            board[r][c] = human
            val, _ = minimax(board, depth - 1, alpha, beta, True, ai, human)
            board[r][c] = 0
            if val < best_val:
                best_val = val
                best_move = (r, c)
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val, best_move


def ai_move(board, ai=2, human=1, depth=SEARCH_DEPTH):
    empty = all(board[r][c] == 0 for r in range(SIZE) for c in range(SIZE))
    if empty:
        return SIZE // 2, SIZE // 2

    # 1) 즉시 5목
    wins = winning_moves(board, ai)
    if wins:
        return pick_best_move(board, wins, ai)

    # 2) 상대 5목 막기
    blocks = winning_moves(board, human)
    if blocks:
        return pick_best_move(board, blocks, ai)

    # 3) 활사 만들기 (상대가 막지 않으면 다음에 5목)
    ai_fours = open_four_moves(board, ai)
    if ai_fours:
        return pick_best_move(board, ai_fours, ai)

    # 4) 상대 활사 막기
    human_fours = open_four_moves(board, human)
    if human_fours:
        return pick_best_move(board, human_fours, ai)

    _, move = minimax(board, depth, -math.inf, math.inf, True, ai, human)
    return move


# ─────────────────────────────────────────
# API
# ─────────────────────────────────────────

class MoveRequest(BaseModel):
    board: List[List[int]]
    row: int
    col: int


class AIResponse(BaseModel):
    row: int
    col: int
    win: Optional[int] = None  # 1=사용자승, 2=AI승, 0=진행중
    forbidden: Optional[bool] = None  # 흑 33 금지


def check_open_three_dir(board, r, c, player, dr, dc):
    """한 방향에서 열린 3 여부 (양 끝 모두 열려있고 연속 3개)"""
    fwd = 0
    nr, nc = r + dr, c + dc
    while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == player:
        fwd += 1
        nr += dr
        nc += dc
    fwd_open = 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == 0

    bwd = 0
    nr, nc = r - dr, c - dc
    while 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == player:
        bwd += 1
        nr -= dr
        nc -= dc
    bwd_open = 0 <= nr < SIZE and 0 <= nc < SIZE and board[nr][nc] == 0

    return (1 + fwd + bwd) == 3 and fwd_open and bwd_open


def count_open_threes(board, r, c, player):
    """해당 위치에 돌을 뒀을 때 생기는 열린 3의 방향 수"""
    board[r][c] = player
    count = sum(1 for dr, dc in DIRECTIONS if check_open_three_dir(board, r, c, player, dr, dc))
    board[r][c] = 0
    return count


@app.post("/move", response_model=AIResponse)
def move(req: MoveRequest):
    board = [row[:] for row in req.board]

    # 흑(player 1) 33 금지 체크
    if count_open_threes(board, req.row, req.col, 1) >= 2:
        return AIResponse(row=-1, col=-1, win=0, forbidden=True)

    # 사용자 돌 놓기
    board[req.row][req.col] = 1

    if check_win(board, 1):
        return AIResponse(row=-1, col=-1, win=1)

    # AI 돌 놓기
    r, c = ai_move(board)
    board[r][c] = 2

    win = 2 if check_win(board, 2) else 0
    return AIResponse(row=r, col=c, win=win)


# ─────────────────────────────────────────
# 프론트엔드 (HTML)
# ─────────────────────────────────────────

HTML = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>오목 — AI vs 나</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a2e; display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; font-family: 'Segoe UI', sans-serif; color: #eee; }
  h1 { font-size: 24px; margin-bottom: 12px; letter-spacing: 2px; color: #e0c97f; }
  #status { font-size: 15px; margin-bottom: 14px; color: #94a3b8; min-height: 22px; }
  canvas { border: 2px solid #e0c97f; border-radius: 4px; cursor: pointer; background: #dcb96b; }
  .controls { display: flex; gap: 12px; margin-top: 16px; }
  .controls button { padding: 10px 28px; background: #e0c97f; color: #1a1a2e; border: none; border-radius: 6px; font-size: 15px; font-weight: 700; cursor: pointer; letter-spacing: 1px; }
  .controls button:hover:not(:disabled) { background: #f0d98f; }
  .controls button:disabled { opacity: 0.4; cursor: not-allowed; }
</style>
</head>
<body>
<h1>오목 — AI vs 나</h1>
<div id="status">당신의 차례 (흑)</div>
<canvas id="board" width="600" height="600"></canvas>
<div class="controls">
  <button id="btn-undo" disabled>한 수 무르기</button>
  <button id="btn-reset">다시 시작</button>
</div>

<script>
const SIZE = 15;
const CELL = 600 / (SIZE + 1);
const R = CELL * 0.38;
const canvas = document.getElementById('board');
const ctx = canvas.getContext('2d');
const status = document.getElementById('status');

let board = Array.from({length: SIZE}, () => Array(SIZE).fill(0));
let gameOver = false;
let myTurn = true;
let thinking = false;
const btnUndo = document.getElementById('btn-undo');

function cloneBoard() {
  return board.map(row => row.slice());
}

let history = [cloneBoard()];

function updateUndoBtn() {
  btnUndo.disabled = thinking || history.length <= 1;
}

function draw() {
  ctx.clearRect(0, 0, 600, 600);
  // 격자
  ctx.strokeStyle = '#8b6914';
  ctx.lineWidth = 1;
  for (let i = 0; i < SIZE; i++) {
    const x = CELL + i * CELL, y = CELL + i * CELL;
    ctx.beginPath(); ctx.moveTo(x, CELL); ctx.lineTo(x, CELL * SIZE); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(CELL, y); ctx.lineTo(CELL * SIZE, y); ctx.stroke();
  }
  // 화점
  const dots = [3, 7, 11];
  ctx.fillStyle = '#8b6914';
  for (const dr of dots) for (const dc of dots) {
    ctx.beginPath();
    ctx.arc(CELL + dc * CELL, CELL + dr * CELL, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  // 돌
  for (let r = 0; r < SIZE; r++) for (let c = 0; c < SIZE; c++) {
    if (!board[r][c]) continue;
    const x = CELL + c * CELL, y = CELL + r * CELL;
    const grad = ctx.createRadialGradient(x - R*0.3, y - R*0.3, R*0.1, x, y, R);
    if (board[r][c] === 1) {
      grad.addColorStop(0, '#555'); grad.addColorStop(1, '#000');
    } else {
      grad.addColorStop(0, '#fff'); grad.addColorStop(1, '#ccc');
    }
    ctx.beginPath();
    ctx.arc(x, y, R, 0, Math.PI * 2);
    ctx.fillStyle = grad;
    ctx.fill();
    ctx.strokeStyle = '#888'; ctx.lineWidth = 0.5; ctx.stroke();
  }
}

function getCell(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const px = (e.clientX - rect.left) * scaleX;
  const py = (e.clientY - rect.top) * scaleY;
  const c = Math.round((px - CELL) / CELL);
  const r = Math.round((py - CELL) / CELL);
  return (r >= 0 && r < SIZE && c >= 0 && c < SIZE) ? {r, c} : null;
}

canvas.addEventListener('click', async (e) => {
  if (!myTurn || gameOver || thinking) return;
  const cell = getCell(e);
  if (!cell || board[cell.r][cell.c]) return;

  board[cell.r][cell.c] = 1;
  myTurn = false;
  thinking = true;
  updateUndoBtn();
  status.textContent = 'AI 생각 중...';
  draw();

  let data;
  try {
    const res = await fetch('/move', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({board, row: cell.r, col: cell.c})
    });
    data = await res.json();
  } catch {
    board[cell.r][cell.c] = 0;
    draw();
    myTurn = true;
    thinking = false;
    updateUndoBtn();
    status.textContent = '오류가 발생했습니다. 다시 시도하세요.';
    return;
  }

  if (data.forbidden) {
    board[cell.r][cell.c] = 0;
    draw();
    myTurn = true;
    thinking = false;
    updateUndoBtn();
    status.textContent = '⛔ 흑 33 금지! 다른 곳에 두세요.';
    return;
  }
  if (data.win === 1) {
    history.push(cloneBoard());
    draw(); status.textContent = '당신이 이겼습니다! 🎉'; gameOver = true;
    thinking = false;
    updateUndoBtn();
    return;
  }
  board[data.row][data.col] = 2;
  draw();
  if (data.win === 2) {
    history.push(cloneBoard());
    status.textContent = 'AI가 이겼습니다 😈'; gameOver = true;
    thinking = false;
    updateUndoBtn();
    return;
  }
  history.push(cloneBoard());
  myTurn = true;
  thinking = false;
  updateUndoBtn();
  status.textContent = '당신의 차례 (흑)';
});

btnUndo.addEventListener('click', () => {
  if (thinking || history.length <= 1) return;
  history.pop();
  board = cloneBoard(history[history.length - 1]);
  gameOver = false;
  myTurn = true;
  status.textContent = '당신의 차례 (흑)';
  draw();
  updateUndoBtn();
});

document.getElementById('btn-reset').addEventListener('click', () => {
  board = Array.from({length: SIZE}, () => Array(SIZE).fill(0));
  history = [cloneBoard()];
  gameOver = false;
  myTurn = true;
  thinking = false;
  status.textContent = '당신의 차례 (흑)';
  draw();
  updateUndoBtn();
});

draw();
updateUndoBtn();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML
