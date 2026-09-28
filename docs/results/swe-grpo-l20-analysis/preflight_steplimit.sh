#!/bin/bash
# 动作预算(agent 动作步数上限)发车前置断言。
#
# 由来:launcher 里的一行
#   `export SWE_AGENT_STEP_LIMIT="${SWE_AGENT_STEP_LIMIT:-15}"`
# 把**冒烟用的 15** 固化成生产默认(代码默认其实是 40),于是整轮评测阶梯
# 6 个点 × 5 片全部跑在 15 步上,而日志、参数 dump、产物**毫无异常**。
# 详见 ../../results/swe-grpo-l20-negative-result.md 的「四处静默失效」①。
#
# 事后实测表明 15→40 这一档对读数影响很小,所以那次没有被毁掉;
# 但「预算被悄悄改小」这件事本身必须**发车就死**,不能靠事后审计。
#
# 用法:  preflight_steplimit.sh <有效值>
#   退出 0 = 放行;退出 1 = 拒绝发车
# 例外:  SWE_ALLOW_SMOKE=1 时允许 < MIN(冒烟必须显式声明自己是冒烟)
#
# 用法取证(而不是"看脚本觉得对"):
#   $ bash preflight_steplimit.sh 15     -> ⛔ 拒绝
#   $ SWE_ALLOW_SMOKE=1 bash ... 15      -> ⚠️ 放行
#   $ bash preflight_steplimit.sh 40     -> ✅ 放行

MIN=40
v="${1:-}"

if [ -z "$v" ]; then
  echo "⛔ 动作预算为空 —— 拒绝发车(空值会退回 launcher 的冒烟默认 15)" >&2
  exit 1
fi

case "$v" in
  ''|*[!0-9]*)
    echo "⛔ 动作预算不是纯正整数: [$v] —— 拒绝发车" >&2
    exit 1
    ;;
esac

if [ "$v" -lt "$MIN" ]; then
  if [ "${SWE_ALLOW_SMOKE:-0}" = "1" ]; then
    echo "⚠️  动作预算 $v < $MIN,但 SWE_ALLOW_SMOKE=1 ⇒ 按冒烟放行"
    exit 0
  fi
  echo "⛔ SWE_AGENT_STEP_LIMIT=$v < $MIN —— 这是冒烟档的值,拒绝发车。" >&2
  echo "   确为冒烟请显式 SWE_ALLOW_SMOKE=1。" >&2
  echo "   本闸由来见 docs/results/swe-grpo-l20-negative-result.md 四处静默失效 ①。" >&2
  exit 1
fi

echo "✅ 动作预算 = $v"
exit 0
