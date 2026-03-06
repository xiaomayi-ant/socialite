# Socialite Agent Contract (v0.1 Draft)

> 目标：把当前“Runner 主导 + 部分消息驱动”演进为“协议驱动的多 Agent 协作系统”。
> 本文档先定义角色边界与消息契约，作为后续重构基线。

## 1. 当前项目审核结论（As-Is）

- 架构现状：`runner.py` 负责主流程编排、提案收集、仲裁后执行、反馈轮询、学习触发、报告触发。
- 消息现状：`collect/analyse` 走 `Message`，但 `propose()/generate_*()` 仍是 Runner 直接方法调用。
- 拓扑现状：已存在 `MsgHub` 订阅关系与 `action_completed` 广播。
- 主要缺口：
  - 缺统一消息 Envelope（缺 `correlation_id/causation_id/ttl/schema_version`）。
  - 缺会话状态机（跨 Agent 交互无明确生命周期）。
  - Agent 注释与实现有漂移（如 Learner 声明会发布 `strategy_update/collection_suggestion`，实现未落地）。

## 2. Agent 角色边界（To-Be）

### 2.1 SensorAgent
- 角色：外部平台 I/O 网关（feed、comment、post、upvote、profile、comments）。
- 负责：
  - 拉取外部数据并标准化。
  - 执行动作命令并回传执行结果事件。
- 不负责：策略判断、内容生成、优先级仲裁。

### 2.2 AnalysisAgent
- 角色：数据增益层（novelty/quality/learning_value/topic/semantic）。
- 负责：对 feed 进行分析，输出结构化分析事件。
- 不负责：直接执行动作。

### 2.3 CommentAgent
- 角色：评论策略与评论内容生成。
- 负责：生成 comment proposal、按批准命令生成 comment 文本。
- 不负责：直接调用平台 API。

### 2.4 PostAgent
- 角色：发帖决策与发帖内容生成。
- 负责：生成 post proposal、按批准命令生成 post 文本。
- 不负责：直接调用平台 API。

### 2.5 UpvoteAgent
- 角色：点赞候选筛选（规则/A-B/LLM）。
- 负责：生成 upvote proposal。
- 不负责：平台执行。

### 2.6 FollowAgent
- 角色：关注候选筛选（互动信号/图谱信号）。
- 负责：生成 follow proposal。
- 不负责：平台执行（当前 API 侧尚未闭环）。

### 2.7 CoordinatorAgent
- 角色：提案仲裁中心。
- 负责：预算检查、配额限制、去重、批准/拒绝。
- 不负责：内容生成与平台 I/O。

### 2.8 LearnerAgent
- 角色：学习与演化控制器。
- 负责：模式挖掘、学习进度评估、演化参数更新、发布策略更新建议。
- 不负责：平台执行。

### 2.9 ObserverAgent
- 角色：审计与观测。
- 负责：A/B 统计、异常检测、周期报告。
- 不负责：业务策略决策。

## 3. Topic 分层与订阅矩阵（To-Be）

原则：按语义订阅，不按对象硬编码调用。

### 3.1 Topic 分层
- `Topic` 不是 Agent，也不是类；它是“消息语义通道”。
- 分层不是“每个 Agent 一层”，而是“按业务阶段/意图分层”。
- Topic 数量不等于 Agent 数量。一个 Agent 可以发布/订阅多个 Topic；一个 Topic 也可以被多个 Agent 消费。
- 关系是多对多（M:N），不是一对一映射。
- `sys.*`：系统控制与健康事件
- `feed.*`：采集与分析
- `proposal.*`：提案
- `coord.*`：仲裁
- `exec.*`：执行
- `feedback.*`：反馈
- `learn.*`：学习与策略
- `observe.*`：观测报告

### 3.2 建议订阅（核心）
- Sensor 订阅：`exec.command.*`, `learn.collection_hint.*`
  - 原因：Sensor 是唯一平台 I/O 执行网关；所有执行命令应收敛到它，学习侧仅给它采集提示。
- Analysis 订阅：`feed.collected`
  - 原因：Analysis 只对“已采集原始数据”负责，输入边界清晰，避免消费执行/学习噪声。
- Comment/Post/Upvote/Follow 订阅：`feed.analyzed`, `learn.strategy_updated`
  - 原因：动作提案要同时依赖“当前上下文（analysis）”和“长期策略（learner）”。
- Coordinator 订阅：`proposal.submitted.*`
  - 原因：仲裁必须是统一入口，才能集中做预算、限额、去重；否则会出现多点决策冲突。
- Learner 订阅：`exec.completed.*`, `feedback.updated`
  - 原因：学习依赖真实执行结果与延迟反馈，不能只看提案或即时信号。
- Observer 订阅：`exec.completed.*`, `coord.decision.*`, `learn.strategy_updated`
  - 原因：观测要覆盖“决策-执行-策略变化”全链路，才能做 A/B 审计和异常归因。

补充：该结构是“最小闭环订阅”，目标是减少无效消费并保留必要双向能力，不是全员互订。

### 3.3 Learner 深化（依赖、输入定义、执行链）

本节回答三个问题：Learner 依赖什么、`执行结果` 是什么、`延迟反馈` 是什么。

#### A. Learner 的核心依赖（当前实现）
- 依赖 1：执行结果流（来自 `exec.completed.*` 对应的广播）
  - 当前落地：Runner 在每次动作后广播 `action_completed`，字段含：
    - `action`, `target_id`, `strategy`, `success`, `priority`, `agent_name`
  - 作用：给 Learner/Observer 提供“动作是否成功”的即时结果。
- 依赖 2：延迟反馈流（来自 `feedback.updated`）
  - 当前落地：Runner 在 Phase 5b 轮询历史评论反馈，按评论发布时间延迟检查。
  - 配置：`FEEDBACK_POLL_DELAYS_HOURS = [0.5, 6, 24, 48]`
  - 作用：弥补“刚发评论时看不出长期效果”的时滞问题。
- 依赖 3：社区样本流（高价值帖子/评论）
  - 当前落地：`get_high_value_posts()`、`get_high_value_comments()`。
  - 作用：用于模式挖掘（不是只看自身行为）。
- 依赖 4：自我行为与身份快照
  - 当前落地：`get_own_comments()` + `get_identity_snapshots()`。
  - 作用：计算验证分、身份一致性分。

#### B. “执行结果”是什么（定义）
- 定义：动作刚执行完得到的即时 outcome（成功/失败及上下文）。
- 当前消息近似结构：
  - `type=action_completed`
  - `action`：`comment|post|upvote|follow`
  - `success`：`true|false`
  - `strategy`：`A|B`
  - `priority`：提案优先级
  - `agent_name`：提案来源 agent
- 特点：快，但信号粗，更多是“执行层成功”，不代表“社区反馈成功”。

#### C. “延迟反馈”是什么（定义）
- 定义：动作执行后，经过一段时间再回看真实互动数据（如评论涨赞）。
- 当前实现（评论反馈）：
  1. 从 `own_comments` 取最近评论。
  2. 根据评论 age 判断是否到达轮询窗口（0.5h/6h/24h/48h）。
  3. 调平台 API 读取对应评论最新 upvotes。
  4. 写回 `own_comments.upvotes_current/upvotes_history/success/feedback_score`。
- 特点：慢，但更接近真实学习信号。

#### D. 当前 Learner 学习执行链（As-Is）
1. Runner 每 `3` 个 cycle 触发一次 `learner.learn()`
2. Learner 进行模式挖掘（社区帖子/评论、讨论深度、自反馈、可选图谱、LLM 抽取）
3. 计算学习进度 `learning_progress`（四项加权）
4. 处理演化（stage、exploration_rate、forgetting）
5. 持久化 `style_patterns` 与 `evolution_state`

### 3.4 Learner 角色与能力（明确化）

- 角色定位：学习控制平面（Learning Control Plane），不是执行平面。
- 能力清单：
  - 模式发现：从多源数据抽取“可复用策略模式”
  - 模式排序：按加权置信度排序并落库
  - 学习评估：输出 0~1 `learning_progress`
  - 演化控制：维护阶段（initial/exploration/optimization/innovation）与探索率
  - 策略建议：在满足条件时生成 `strategy_notes/adjustments`
- 非目标：
  - 不直接执行平台动作
  - 不直接决定单条提案是否通过（那是 Coordinator 责任）

### 3.5 对当前 Learner 学习方式的评估

### 优点
- 有“即时结果 + 延迟反馈 + 社区样本”三类数据源，方向正确。
- 学习评分拆解为四维（depth/validation/consistency/uniqueness），具备可解释性。
- 阶段化探索率策略（初始/探索/优化/创新）有利于长期探索。

### 局限（当前代码层面）
- `action_completed` 已进入 `_action_buffer`，但学习主链未消费该 buffer，信号未充分利用。
- `performance_data` 当前由 Runner 传空数组，导致演化指标长期弱信号：
  - `_get_strategy_advice()`几乎不会触发（需要非空 performance_data）。
- 身份一致性计算依赖 `core_values/default_positions`，但当前快照主要是 `topic/stance`，一致性分辨率有限。
- 延迟反馈目前主要覆盖评论 upvote，帖子/关注等动作反馈闭环较弱。
- 部分 pattern 字段（如 `weighted_confidence`）在持久化层未完整落库，损失排序信息。

### 结论（是否可用）
- 现状属于“可运行的学习雏形”，可驱动基础迭代；
- 但若要成为高质量策略引擎，需先补齐：
  - 执行结果信号入学习链
  - 完整 performance_data 管道
  - 多动作延迟反馈闭环
  - 模式字段与评分口径统一

## 4. 消息 Envelope 规范（必填）

```json
{
  "message_id": "uuid",
  "message_type": "event|command|query|reply",
  "topic": "proposal.submitted.comment",
  "source_agent": "comment",
  "target_role": "coordinator",
  "target_agent": null,
  "trace_id": "uuid",
  "correlation_id": "uuid",
  "causation_id": "uuid|null",
  "cycle_id": 42,
  "priority": 0.7,
  "ttl_seconds": 120,
  "schema_version": "1.0",
  "created_at": "2026-03-03T21:00:00Z",
  "payload": {}
}
```

说明：
- `target_role` 用于逻辑定向（如 coordinator），避免发送方硬编码具体实例。
- `correlation_id` 用于串联同一会话；`causation_id` 用于追溯触发关系。
- `ttl_seconds` 防止陈旧消息被继续消费。

## 5. 会话状态机（Saga/Conversation）

## 5.1 会话 A：FeedAnalysis
1. `feed.collect.requested` (command)
2. `feed.collected` (event)
3. `feed.analyze.requested` (command)
4. `feed.analyzed` (event)
5. `proposal.submitted.*` (event, by action agents)
6. `conversation.closed` or `conversation.timeout`

## 5.2 会话 B：ActionExecution
1. `proposal.submitted.<action>`
2. `coord.decision.approved|rejected`
3. `exec.command.<action>` (approved only)
4. `exec.completed.<action>|exec.failed.<action>`
5. `feedback.requested` -> `feedback.updated`
6. `conversation.closed`

## 5.3 会话 C：LearningLoop
1. `learn.triggered`
2. `learn.patterns.mined`
3. `learn.progress.evaluated`
4. `learn.strategy_updated`
5. `learn.collection_hint.updated`
6. `conversation.closed`

## 6. Agent 决策与回复规则

- Event：默认不强制回复，按订阅策略消费。
- Command：必须 `ack|nack`，并在超时窗口内完成。
- Query：必须回复 `reply`（可带 `not_available`）。
- Reply：只能响应对应 `correlation_id` 的未完成请求。
- 任一 Agent 可选择“不继续对话”，但必须给出可机读原因（`nack_reason`）。

## 7. 可靠性与安全约束（必须）

- Topic 分层，禁止单一全局广播作为常规路径。
- Handler 幂等（按 `message_id` 去重，副作用操作需幂等键）。
- 投递语义：至少一次 + 指数退避重试 + DLQ。
- 会话级超时、最大跳数（`max_hops`）防环路。
- Agent 并发上限（in-flight）与背压（队列长度阈值）。

## 8. 与当前代码的映射差距（实施清单）

- 将 Runner 直接调用替换为消息：
  - `comment.propose()` -> `proposal.submitted.comment`
  - `post.generate_post()` -> `exec.command.post_generate` / `exec.command.post_publish`
- 在 `core.message.Message` 扩展 Envelope 字段或新增 `EnvelopeMessage`。
- 在 `MsgHub` 增加：
  - Topic 路由
  - `ack/nack` 通道
  - 去重缓存与 TTL 检查
  - DLQ 转发
- 增加 `ConversationManager`：
  - 管理 `correlation_id` 状态机
  - 超时关闭与补偿逻辑

## 9. Prompt 优化方向（先记录，不在本次实现）

- Analysis prompt：输出字段增加置信度与“不确定性原因”。
- Comment/Post prompt：加入结构化 `style_guardrails`，减少模板化表达。
- Learner prompt：限制泛化结论，要求 pattern 必须绑定可验证指标。
- 所有 LLM JSON 输出统一 schema + strict parser + fallback contract。

---

本文件是 Socialite 的 Agent 协作契约草案，后续以代码实现为准并保持版本化更新。
