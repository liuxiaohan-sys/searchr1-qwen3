"""Async verl Agent Loop adapter for Search-R1's XML action protocol."""

import asyncio
import json
import re
from typing import Any
from uuid import uuid4

import requests

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput


class SearchR1AgentLoop(AgentLoopBase):
    """Run one Search-R1 trajectory on verl's request-level async rollout stack.

    The policy keeps using the protocol it was trained on:
    ``<search>...</search>`` requests retrieval and
    ``<answer>...</answer>`` terminates the trajectory.
    """

    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        cls.tokenizer = tokenizer
        cls.max_turns = int(config.max_turns)
        cls.max_start_length = int(config.data.max_start_length)
        cls.response_length = int(config.actor_rollout_ref.rollout.response_length)
        cls.turn_response_length = int(config.data.max_response_length)
        cls.search_url = config.retriever.url
        cls.topk = int(config.retriever.topk)
        cls.search_timeout = float(kwargs.get("search_timeout", 120.0))

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        # Match the legacy Search-R1 loop, which keeps only the configured
        # starting context before appending generated turns and observations.
        prompt_ids = prompt_ids[-self.max_start_length :]
        original_prompt_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        request_id = uuid4().hex
        search_count = 0
        invalid_action_count = 0
        assistant_turns = 0

        # max_turns is the number of retrieval rounds. The extra generation is
        # the final-answer round, matching LLMGenerationManager.run_llm_loop().
        for turn in range(self.max_turns + 1):
            if len(response_ids) >= self.response_length:
                break

            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = min(
                self.turn_response_length,
                self.response_length - len(response_ids),
            )
            output = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=turn_sampling_params,
            )
            assistant_turns += 1

            generated_ids, generated_text = self._truncate_at_action(output.token_ids)
            generated_ids = generated_ids[: self.response_length - len(response_ids)]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            response_ids.extend(generated_ids)
            response_mask.extend([1] * len(generated_ids))
            prompt_ids.extend(generated_ids)

            action, content = self._parse_action(generated_text)
            if action == "answer":
                break

            if action == "search" and turn < self.max_turns:
                search_count += 1
                search_result = await asyncio.to_thread(self._search, content)
                turn_number = turn + 1
                final_hint = ""
                if turn_number >= self.max_turns:
                    final_hint = " 注意：接下来必须总结已有信息并给出最终回答！"
                observation = (
                    f"\n[系统提示] 当前为第 {turn_number} 轮检索（上限 {self.max_turns} 轮）。"
                    f"{final_hint}\n<information>{search_result}</information>\n\n"
                )
            elif action == "search":
                observation = "\n接下来必须总结已有信息，并在 <answer> 和 </answer> 中给出最终回答。\n"
            else:
                invalid_action_count += 1
                observation = (
                    "\n【系统提示】如果需要搜索，请把合法 JSON 放在 <search> 和 </search> 之间；"
                    "如果给出最终回答，请把答案放在 <answer> 和 </answer> 之间。请重新输出。【/系统提示】\n"
                )

            observation_ids = self.tokenizer.encode(observation, add_special_tokens=False)
            remaining = self.response_length - len(response_ids)
            if remaining <= 0:
                break
            observation_ids = observation_ids[:remaining]
            response_ids.extend(observation_ids)
            # Environment observations must not contribute to the policy loss.
            response_mask.extend([0] * len(observation_ids))
            prompt_ids.extend(observation_ids)

        return AgentLoopOutput(
            prompt_ids=original_prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            num_turns=assistant_turns + search_count,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "search_count": search_count,
                "invalid_action_count": invalid_action_count,
            },
        )

    def _truncate_at_action(self, token_ids: list[int]) -> tuple[list[int], str]:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        search_end = text.find("</search>")
        answer_end = text.find("</answer>")
        candidates = []
        if search_end >= 0:
            candidates.append(search_end + len("</search>"))
        if answer_end >= 0:
            candidates.append(answer_end + len("</answer>"))
        if candidates:
            text = text[: min(candidates)]
            # This mirrors the legacy Search-R1 loop, which also decodes,
            # truncates at the action boundary, and tokenizes again.
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return list(token_ids), text

    @staticmethod
    def _parse_action(text: str) -> tuple[str, str]:
        match = re.search(r"<(search|answer)>(.*?)</\1>", text, re.DOTALL)
        if match is None:
            return "invalid", ""
        return match.group(1), match.group(2).strip()

    def _search(self, search_json: str) -> str:
        try:
            clean = search_json.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip().replace("，", ",").replace("“", '"').replace("”", '"')
            if not clean.startswith("{"):
                clean = "{" + clean
            if not clean.endswith("}"):
                clean += "}"

            payload = {"query": json.loads(clean), "topk": self.topk}
            response = requests.post(
                self.search_url,
                json=payload,
                proxies={"http": None, "https": None},
                timeout=self.search_timeout,
            )
            response.raise_for_status()
            return self._format_search_response(response.json())
        except json.JSONDecodeError:
            return "工具调用失败：<search> 标签内的 JSON 格式不合法，请修正后重试。"
        except Exception as exc:
            return f"检索请求出错：{exc}"

    @staticmethod
    def _format_search_response(data: dict[str, Any]) -> str:
        if "error" in data:
            return f"检索返回错误：{data['error']}"

        request_type = data.get("检索类型", "")
        if request_type == "类案检索" or "llm_summary" in data:
            summary = data.get("llm_summary", "未检索到匹配的类案分析结果。")
            return f"【类案检索分析报告】\n{summary}"

        results = data.get("result")
        if request_type == "法律检索" or isinstance(results, list):
            references = []
            for index, item in enumerate(results or [], start=1):
                content = item.get("document", {}).get("content", "")
                score = float(item.get("score", 0.0))
                references.append(f"法律参考 {index}（相关度 {score:.4f}）：\n{content}\n")
            if references:
                return "【法律检索结果】\n" + "\n".join(references)
            return "【法律检索结果】未找到相关法律条文，请更换关键词。"

        return f"未知的检索返回类型，原始数据：{str(data)[:200]}"
