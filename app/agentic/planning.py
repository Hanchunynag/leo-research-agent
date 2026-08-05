"""可解释、类别感知的科研 Query Planner。"""

from __future__ import annotations

from typing import Literal

from app.agentic.models import (
    EvidenceRequirement,
    PlannedSubquestion,
    QueryPlan,
)
from app.indexing.tokenization import normalize_search_text


class QueryPlanner:
    """区分观测量、输入、先验、状态、参数、方法、结果与指标。"""

    def plan(self, standalone_query: str) -> QueryPlan:
        query = standalone_query.strip()
        if not query:
            raise ValueError("standalone_query 不能为空。")
        normalized = normalize_search_text(query)
        intent = self._intent(normalized)
        evolution = self._is_evolution_query(normalized)
        target = "method" if evolution else self._target_category(normalized)
        excluded = (
            ["input", "prior", "state", "parameter", "method", "result"]
            if target in {"measurement", "observable"}
            else []
        )
        subquestions = self._subquestions(query, normalized)
        retrieval_queries = list(
            dict.fromkeys([query, *(item.question for item in subquestions)])
        )
        if evolution:
            retrieval_queries.extend(
                [
                    "proposed method contribution abstract conclusion",
                    "method improvement extension experimental validation",
                ]
            )
            retrieval_queries = list(dict.fromkeys(retrieval_queries))
            requirements = [
                EvidenceRequirement(
                    subquestion_id=item.id,
                    requirement=(
                        "使用多篇论文对各自方法、贡献或扩展的直接证据"
                        "组合支持该子问题；不要求任何单篇论文显式总结"
                        "整个跨论文演进路线。"
                    ),
                )
                for item in subquestions
            ]
        else:
            requirements = [
                EvidenceRequirement(
                    subquestion_id=item.id,
                    requirement=(
                        f"直接说明 {target} 被用于该任务的正文证据，"
                        "不能只有背景相关性。"
                    ),
                )
                for item in subquestions
            ]
        constraints = [
            f"直接答案只允许列出 category={target} 的对象。",
            "不同论文或实验条件必须分别说明。",
            "不得用模型常识补充本地证据未提供的信息。",
        ]
        if evolution:
            constraints.extend(
                [
                    "演进路线是跨论文综合：每个阶段必须分别引用对应"
                    "论文的方法或结果证据。",
                    "跨论文的“从 A 到 B”比较必须同时引用 A 和 B 的"
                    "证据，不得声称某篇论文自身提出了整体时间线。",
                    "如发表年份不足以确定严格时序，应按可验证的"
                    "技术递进关系组织，并明确说明。",
                ]
            )
        if target in {"measurement", "observable"}:
            constraints.extend(
                [
                    "预测星历、SGP4 传播结果属于辅助输入或先验，不得列为观测量。",
                    "载波相位、多普勒和伪距只有在正文明确使用时才能列为观测量。",
                ]
            )
        return QueryPlan(
            intent=intent,
            target_category=target,
            excluded_categories=excluded,
            subquestions=subquestions,
            retrieval_queries=retrieval_queries,
            required_evidence=requirements,
            answer_constraints=constraints,
        )

    @staticmethod
    def _intent(
        normalized: str,
    ) -> Literal[
        "fact_list",
        "definition",
        "mechanism",
        "comparison",
        "method",
        "numeric_result",
        "citation_lookup",
        "synthesis",
    ]:
        if any(value in normalized for value in ("为什么", "how", "mechanism")):
            return "mechanism"
        if any(value in normalized for value in ("比较", "区别", "versus", "compare")):
            return "comparison"
        if any(value in normalized for value in ("多少", "数值", "accuracy", "error")):
            return "numeric_result"
        if any(value in normalized for value in ("哪些", "what observations", "list")):
            return "fact_list"
        if any(value in normalized for value in ("是什么", "定义", "what is")):
            return "definition"
        return "synthesis"

    @staticmethod
    def _target_category(normalized: str) -> str:
        mappings = (
            (("观测量", "观测", "measurement", "observable"), "measurement"),
            (("输入", "input"), "input"),
            (("先验", "prior"), "prior"),
            (("状态", "state"), "state"),
            (("参数", "parameter"), "parameter"),
            (("指标", "metric"), "metric"),
            (("数据集", "dataset"), "dataset"),
            (("假设", "assumption"), "assumption"),
            (("结果", "result"), "result"),
            (("方法", "算法", "method"), "method"),
        )
        for terms, category in mappings:
            if any(term in normalized for term in terms):
                return category
        return "other"

    @staticmethod
    def _is_evolution_query(normalized: str) -> bool:
        """识别需要跨文献组合证据的演进、路线或时间线问题。"""

        return any(
            marker in normalized
            for marker in (
                "演进",
                "演化",
                "发展路线",
                "研究路线",
                "技术路线",
                "时间线",
                "evolution",
                "progression",
                "roadmap",
                "timeline",
            )
        )

    @staticmethod
    def _subquestions(query: str, normalized: str) -> list[PlannedSubquestion]:
        if QueryPlanner._is_evolution_query(normalized):
            return [
                PlannedSubquestion(
                    id="SQ1",
                    question="各篇相关论文分别提出了什么核心方法或技术贡献？",
                ),
                PlannedSubquestion(
                    id="SQ2",
                    question=(
                        "这些方法在观测信息、误差建模和状态估计策略上"
                        "有哪些可由证据支持的技术递进？"
                    ),
                ),
                PlannedSubquestion(
                    id="SQ3",
                    question=(
                        "后续工作增加了哪些联合估计、预测、实验验证"
                        "或导航应用扩展？"
                    ),
                ),
            ]
        if "星历" in normalized and any(
            term in normalized for term in ("时钟", "钟漂", "clock")
        ):
            return [
                PlannedSubquestion(
                    id="SQ1",
                    question="哪些导航观测量被直接用于估计低轨卫星星历误差？",
                ),
                PlannedSubquestion(
                    id="SQ2",
                    question="哪些导航观测量被直接用于估计时钟偏差或时钟漂移？",
                ),
            ]
        return [PlannedSubquestion(id="SQ1", question=query)]
