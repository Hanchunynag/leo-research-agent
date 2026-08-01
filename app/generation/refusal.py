"""生成链路中可预测、失败关闭的拒答原因。"""

EMPTY_CONTEXT_REFUSAL = "当前检索证据为空，无法生成有依据的回答。"
INVALID_CONTEXT_REFUSAL = "证据上下文完整性校验失败，已拒绝生成回答。"
INVALID_DRAFT_REFUSAL = "模型返回的回答未通过逐条引用校验，已拒绝输出。"
PROVIDER_ERROR_REFUSAL = "回答模型调用失败，无法生成有依据的回答。"
