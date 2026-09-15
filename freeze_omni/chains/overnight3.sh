#!/bin/bash
# 链 3（链 2 之后）：训练分布内诊断 + 杠杆探针
cd ~/freeze_omni
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
FDB=~/freeze_omni/data/fdb/v1.0/unz
until [ -f logs/FO_NIGHT2_DONE ]; do sleep 120; done
# ① 训练分布内：candor_turn_taking 前 40 条，加噪，冻结前端 vs soft_ann 前端（与 fo_fdb_ann 目标比）
mkdir -p data/fdb_probe/candor_turn_taking; ls -d $FDB/candor_turn_taking/*/ | head -40 | while read d; do ln -sfn "$d" data/fdb_probe/candor_turn_taking/$(basename $d); done
./venv/bin/python -u fo_placement.py --fdb --humdial data/fdb_probe --out nz/probe_fdb_pre --noise-dir data/demand --rnnoise final.pt --max-sec 12 > logs/probe_fdb_pre.log 2>&1
./venv/bin/python -u fo_placement.py --fdb --humdial data/fdb_probe --out nz/probe_fdb_sann --noise-dir data/demand --rnnoise exp/fo_s_ann/epoch_002.pt --max-sec 12 > logs/probe_fdb_sann.log 2>&1
# ② 杠杆探针（HumDial 4 场景×10，干净）：-10dB / -20dB / 低通 1kHz
for cfg in "gain-db -10:g10" "gain-db -20:g20" "lowpass 1000:lp1k"; do
  arg=${cfg%%:*}; tag=${cfg##*:}
  ./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 10 --out nz/probe_$tag --$arg > logs/probe_$tag.log 2>&1
done
for t in probe_g10 probe_g20 probe_lp1k; do echo "== $t"; ./venv/bin/python humdial_metrics3.py --root nz/$t --humdial $HD 2>&1 | tail -2; done > logs/fo_tables_probe.txt
echo DONE > logs/FO_NIGHT3_DONE
# 追加：ES 最终 θ（last.pt）的评测——best.pt 是随机批次下的运气最优，last.pt 才是优化终点
HD=~/freeze_omni/data/humdial/unz/test/en_test_nondev
./venv/bin/python -u fo_placement.py --humdial $HD --per-scen 20 --out nz/fo_rl_last --noise-dir data/demand --rnnoise final.pt --policy exp/fo_rl_ann/last.pt > logs/fo_eval_rl_last.log 2>&1
./venv/bin/python humdial_metrics3.py --root nz/fo_rl_last --humdial $HD 2>&1 | tail -3 >> logs/fo_tables_probe.txt
echo DONE > logs/FO_NIGHT4_DONE
