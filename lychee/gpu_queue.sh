#!/usr/bin/env bash
# GPU 作业队列守护：一个作业跑完立刻接下一个，**不等人**。
#
# 起因（2026-08-20）：新机器上线后 GPU 空转了数小时——不是因为没活干，而是每个作业结束/失败
# 都要等我下一轮交互才能推进。作业本身只占 2 小时，人的往返占了另外 3 小时。
#
#     bash scripts/gpu_queue.sh start      # 起守护（幂等，已在跑就不重复起）
#     bash scripts/gpu_queue.sh status     # 看队列与进度
#     bash scripts/gpu_queue.sh stop
#
# 作业 = $QDIR/NN-名字.sh，按文件名排序执行。跑完移到 done/，失败移到 failed/。
# **随时可以往 $QDIR 丢新作业**，守护会在当前作业结束后自动捡起来——这是"不空闲"的关键。
#
# 作业脚本的退出码约定：
#     0   成功        → 移到 done/
#     75  前置未就绪  → 移回队尾，先跑后面的（EX_TEMPFAIL，避免一个卡住全队）
#     其他 失败       → 移到 failed/，继续下一个（**不停队**）
#
# 每个作业脚本自己负责：激活环境、取 gpu_lock、检查前置。守护只管调度和记账。
set -uo pipefail

WORK="${WORK:-/ephemeral/work}"
QDIR="${QDIR:-$WORK/queue}"
LOG="$WORK/logs/gpu_queue.log"
PIDF="/tmp/.gpu_queue.pid"
mkdir -p "$QDIR" "$QDIR/done" "$QDIR/failed" "$WORK/logs"

say() { printf '[%s] %s\n' "$(date +'%m-%d %H:%M:%S')" "$*" | tee -a "$LOG"; }

run_loop() {
  say "=== 队列守护启动 (pid $$) ==="
  local idle_since=0
  while :; do
    # 取队列里第一个 .sh（按名字排序；done/failed 是子目录，-maxdepth 1 不会捡到）
    local job
    job=$(find "$QDIR" -maxdepth 1 -name '*.sh' -type f 2>/dev/null | sort | head -1)
    if [[ -z "$job" ]]; then
      if (( idle_since == 0 )); then idle_since=$(date +%s); say "队列空，待命中（丢新作业进 $QDIR 即可）"; fi
      sleep 60; continue
    fi
    idle_since=0
    local name; name=$(basename "$job")
    local jlog="$WORK/logs/job_${name%.sh}.log"
    say "▶ 开始 $name  （日志 $jlog）"
    local t0=$(date +%s)
    bash "$job" > "$jlog" 2>&1
    local rc=$?
    local dt=$(( ($(date +%s) - t0) / 60 ))
    case $rc in
      0)  mv "$job" "$QDIR/done/";   say "✓ 完成 $name（${dt} min）" ;;
      75) # 前置未就绪：改名沉到队尾，先跑后面的
          local base=${name#99-deferred-*-}; local newname; newname="99-deferred-$(date +%H%M%S)-$base"
          mv "$job" "$QDIR/$newname"; say "⏸ $name 前置未就绪，沉到队尾（${dt} min）→ $newname"
          sleep 30 ;;
      *)  mv "$job" "$QDIR/failed/"; say "✗ 失败 $name（退出码 $rc，${dt} min）—— 队列继续，不停" ;;
    esac
  done
}

case "${1:-status}" in
  start)
    if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      say "守护已在运行 (pid $(cat "$PIDF"))"; exit 0
    fi
    setsid nohup bash "$0" _run > /dev/null 2>&1 &
    sleep 2; say "守护已起"
    ;;
  _run) echo $$ > "$PIDF"; trap 'rm -f "$PIDF"' EXIT; run_loop ;;
  stop)
    [[ -f "$PIDF" ]] && { kill "$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; say "已停"; } || say "未在运行"
    ;;
  status)
    if [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then echo "守护: 运行中 (pid $(cat "$PIDF"))"; else echo "守护: 未运行"; fi
    echo "--- 待跑 ---"; find "$QDIR" -maxdepth 1 -name '*.sh' | sort | sed 's|.*/|  |' || true
    echo "--- 已完成 ---"; ls "$QDIR/done" 2>/dev/null | sed 's|^|  |'
    echo "--- 失败 ---"; ls "$QDIR/failed" 2>/dev/null | sed 's|^|  |'
    echo "--- 最近日志 ---"; tail -12 "$LOG" 2>/dev/null
    ;;
  *) echo "用法: $0 {start|stop|status}"; exit 1 ;;
esac
