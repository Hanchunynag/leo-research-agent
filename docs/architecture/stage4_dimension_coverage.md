# Stage 4 Dimension Coverage

`app.research.coverage.QueryFrame` 包含：

- `research_object`
- `current_method`
- `target_outcome`
- `required_dimensions`
- `optional_dimensions`
- `excluded_dimensions`

`QueryFrameBuilder` 首先使用中英文规则词典。对“伪距修正星历后，如何通过伪距率、自适应噪声和鲁棒权值提高定位精度？”会得到：

```text
ephemeris_correction
range_rate_observation
state_observability
adaptive_noise_estimation
robust_weighting
positioning_accuracy
```

`DimensionCoverageAnalyzer` 从 Evidence 的正文、`dimension_tags`、实体、Relation 标签和 `relation_path` 计算：Required Dimension Coverage、Direct Evidence Coverage、Source Diversity、Conflict Coverage、Relation Path Coverage 和 Missing Dimensions。

`RelationReasoningWorkflow` 保留“至少两条 Evidence”的最低保护，但充分性由研究维度报告决定。仅在缺失维度存在时执行一次限定补检索，查询中明确包含 `missing_dimensions`；不会默认调用强模型 Planner。测试位于
`tests/test_stage4_dimension_coverage.py`。

待确认：规则词典当前聚焦 LEO 导航、星历、伪距率、VCE 和鲁棒估计。其他科研领域需要增加受控词典或注入一次小型结构化模型，不能静默猜测维度。
