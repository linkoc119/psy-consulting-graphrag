"""
Prompt builder for GraphRAG response generation.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from config import settings
from ..utils.prompts import (
    SYSTEM_PROMPT_COUNSELING,
    SYSTEM_PROMPT_CRISIS,
    MICRO_FEW_SHOT_COUNSELING,
    MICRO_FEW_SHOT_CRISIS,
)

TRIAGE_THRESHOLD_HIGH = settings.TRIAGE_THRESHOLD_HIGH

logger = logging.getLogger(__name__)


def _get_message_field(message: Any, field: str, default: str = "") -> str:
    if isinstance(message, dict):
        return message.get(field, default)
    return getattr(message, field, default)


def _format_context(context: Dict[str, Any]) -> str:
    documents = context["documents"]
    graph_nodes = context["graph_nodes"]
    relationships = context.get("relationships", [])

    context_parts = []

    if documents:
        context_parts.append("=== TÀI LIỆU THAM KHẢO ===")
        for i, doc in enumerate(documents[:3], 1):
            meta = doc["metadata"]
            doc_type = meta.get("doc_type", "")
            context_parts.append(f"[{i}] ({doc_type}) {doc['text'][:300]}...")

    if graph_nodes:
        context_parts.append("\n=== CÁC KHÁI NIỆM TRI THỨC LIÊN QUAN ===")
        for i, node in enumerate(graph_nodes[:3], 1):
            node_type = node.get("metadata", {}).get("node_type", "Unknown")
            context_parts.append(f"[{i}] ({node_type}) {node['text'][:200]}...")

    if relationships:
        context_parts.append("\n=== CÁC MỐI LIÊN KẾT TRI THỨC ===")
        context_parts.append("(Các mối liên hệ quan trọng):")
        for i, rel in enumerate(relationships[:10], 1):
            context_parts.append(f"  {i}. {rel}")

    return "\n".join(context_parts) if context_parts else "Không có ngữ cảnh đặc biệt."


def _format_history(history: Optional[List[Dict]]) -> str:
    history_parts = []
    if history:
        for msg in history[-8:]:
            role = _get_message_field(msg, "role", "user")
            content = _get_message_field(msg, "content", "")
            if role != "user" or not content:
                continue
            history_parts.append(f"Người dùng: {content}")

    return "\n".join(history_parts) if history_parts else "Chưa có lịch sử hội thoại."


def _format_known_facts(history: Optional[List[Dict]]) -> Tuple[str, int]:
    known_facts_parts = []
    if history:
        for msg in history[-6:]:
            role = _get_message_field(msg, "role", "user")
            content = _get_message_field(msg, "content", "")
            if role == "user" and content:
                known_facts_parts.append(f"- {content}")

    known_facts = "\n".join(known_facts_parts) if known_facts_parts else "- Chưa có dữ kiện từ lượt trước."
    user_turn_count = len(known_facts_parts) + 1
    return known_facts, user_turn_count


def _has_time_constraint(query_lower: str) -> bool:
    time_words = [
        "hôm nay", "tối nay", "sáng nay", "chiều nay", "ngày mai", "tuần này",
        "chỉ còn", "còn khoảng", "trong vòng", "trước khi", "sắp phải"
    ]
    if any(word in query_lower for word in time_words):
        return True
    return bool(re.search(r"\b\d+\s*(phút|tiếng|giờ|ngày|tuần|tháng)\b", query_lower))


def _has_action_constraint(query_lower: str) -> bool:
    constraint_words = [
        "khó nhất", "lo nhất", "yếu nhất", "ưu tiên", "bắt đầu", "tập trung vào",
        "chỉ muốn", "chỉ cần", "không muốn", "không thể", "hay sai", "hay quên",
        "còn nhiều", "quá dài", "quá rối", "cụ thể", "rõ ràng"
    ]
    task_words = [
        "môn", "phần", "bài", "việc", "chủ đề", "dạng", "kỹ năng", "mục tiêu",
        "kế hoạch", "lịch", "phiên", "bài tập", "cách làm"
    ]
    return any(word in query_lower for word in constraint_words) or any(word in query_lower for word in task_words)


def _build_turn_guidance(query: str, severity: int, user_turn_count: int) -> str:
    advice_keywords = [
        "gợi ý", "nên làm gì", "làm gì tiếp", "cách", "phương pháp",
        "kế hoạch", "giải pháp", "hướng dẫn", "giúp tôi", "cụ thể"
    ]
    plan_keywords = [
        "kế hoạch", "lịch", "mỗi phiên", "bao lâu", "nghỉ bao lâu",
        "dạng bài", "dừng học", "mấy giờ", "vài ngày", "từng bước",
        "làm gì trước", "chia nhỏ", "thời gian"
    ]
    revision_keywords = [
        "viết lại", "sửa lại", "chỉnh", "rối", "khó đọc", "rõ ràng",
        "từng dòng", "đừng hỏi", "không cần hỏi", "không hỏi", "đưa luôn"
    ]
    no_question_keywords = ["đừng hỏi", "không cần hỏi", "không hỏi", "đưa luôn"]
    query_lower = query.lower()
    asks_for_advice = any(keyword in query_lower for keyword in advice_keywords)
    asks_for_plan = any(keyword in query_lower for keyword in plan_keywords)
    has_time_constraint = _has_time_constraint(query_lower)
    has_action_constraint = _has_action_constraint(query_lower)
    asks_for_revision = any(keyword in query_lower for keyword in revision_keywords)
    no_more_questions = any(keyword in query_lower for keyword in no_question_keywords)
    asks_for_line_format = any(keyword in query_lower for keyword in ["từng dòng", "rõ ràng", "khó đọc"])
    avoid_made_up_examples = any(keyword in query_lower for keyword in ["không tự bịa", "không bịa", "chỉ ghi dạng"])
    has_prior_context = user_turn_count >= 2
    needs_immediate_action = has_prior_context and has_time_constraint and has_action_constraint
    has_enough_context = asks_for_advice or user_turn_count >= 3

    if severity >= TRIAGE_THRESHOLD_HIGH:
        return (
            "Ưu tiên an toàn ngay. Hỏi ngắn về mức độ an toàn hiện tại và hướng người dùng kết nối ngay với người lớn tin cậy hoặc dịch vụ khẩn cấp."
        )

    if asks_for_revision:
        guidance_parts = [
            "Người dùng đang yêu cầu chỉnh/sửa câu trả lời trước. PHẢI viết lại theo yêu cầu mới, không lặp nguyên cấu trúc hoặc ví dụ cũ.",
            "Ưu tiên làm đúng ràng buộc mới nhất của người dùng hơn việc hỏi thêm."
        ]
        if asks_for_line_format:
            guidance_parts.append("Trình bày rõ ràng theo từng dòng hoặc từng mốc thời gian; không dồn nhiều mốc vào một đoạn.")
        if avoid_made_up_examples:
            guidance_parts.append("Không tự bịa đề bài, tên bài, sự kiện hoặc dữ kiện cụ thể; chỉ ghi dạng việc/bài cần làm và cách tự kiểm tra.")
        if no_more_questions:
            guidance_parts.append("Không hỏi xác nhận, không hỏi thêm bối cảnh, và không kết thúc bằng câu hỏi.")
        return " ".join(guidance_parts)

    if needs_immediate_action:
        return (
            "Người dùng đã có bối cảnh trước đó và hiện đã nêu ràng buộc hành động khá cụ thể như thời gian, phạm vi, việc ưu tiên hoặc điều không muốn làm. "
            "PHẢI chuyển từ hỏi làm rõ sang hỗ trợ hành động nhỏ ngay; không hỏi 'bạn muốn điều chỉnh thế nào', không hỏi 'nên bắt đầu từ đâu', không hỏi đã thử cách nào khác. "
            "Bám vào đúng ràng buộc người dùng vừa nêu và đưa một kế hoạch/bước tiếp theo ngắn, khả thi trong phạm vi đó. "
            "Nếu phù hợp, chia thành 3-5 mốc hoặc bước nhỏ; mỗi bước ghi việc cần làm và cách tự kiểm tra. "
            "Không tự bịa ví dụ, đề bài, sự kiện hoặc dữ kiện cụ thể ngoài lời người dùng. "
            "Kết thúc bằng một câu chốt ngắn, không kết thúc bằng câu hỏi."
        )

    if asks_for_plan:
        closing_rule = (
            "Không hỏi xác nhận, không hỏi thêm bối cảnh, và không kết thúc bằng câu hỏi."
            if no_more_questions
            else "Kết thúc bằng tối đa 1 câu hỏi chọn lựa rất ngắn, không hỏi 'bạn thấy thế nào' chung chung."
        )
        format_rule = (
            "Trình bày mỗi mốc/phiên trên một dòng riêng, dễ đọc."
            if asks_for_line_format
            else "Trình bày bằng các mục ngắn theo ngày hoặc theo phiên."
        )
        example_rule = (
            "Không tự bịa đề bài cụ thể; chỉ ghi dạng bài cần làm và cách tự kiểm tra."
            if avoid_made_up_examples
            else ""
        )
        return (
            "Người dùng đang yêu cầu một kế hoạch/hướng dẫn cụ thể. "
            "PHẢI đưa kế hoạch hành động ngay, không hỏi xác nhận và không hỏi thêm bối cảnh. "
            "Bám sát mọi ràng buộc người dùng nêu như số ngày, thời lượng học, thời gian nghỉ, dạng việc/bài tập, thời điểm dừng, chủ đề ưu tiên. "
            f"{format_rule} "
            "Mỗi phiên nên có: mục tiêu, thời lượng, việc cần làm, nghỉ bao lâu, và cách tự kiểm tra. "
            f"{example_rule} "
            f"{closing_rule}"
        )

    if has_enough_context:
        return (
            "Lượt này đã đủ bối cảnh hoặc người dùng đang yêu cầu gợi ý. "
            "PHẢI đưa 2-3 gợi ý cụ thể, nhỏ và khả thi dựa trên dữ kiện đã có. "
            "KHÔNG hỏi lại người dùng đã thử gì, kéo dài bao lâu, hoặc điều gì có thể giúp họ. "
            "KHÔNG hỏi 'bạn nghĩ nên bắt đầu từ đâu' nếu người dùng đã nêu môn/chủ đề hoặc thời lượng. "
            "Chỉ được kết bằng tối đa 1 câu hỏi chọn lựa để người dùng chọn bước muốn thử."
        )

    if has_prior_context:
        return (
            "Đây không phải lượt đầu. Không mở đầu bằng cách lặp lại nguyên vấn đề chung từ lượt trước. "
            "Chỉ phản ánh 1-2 chi tiết mới trong câu hiện tại của người dùng, ví dụ cảm xúc mới, hành vi mới, thời điểm mới hoặc khó khăn mới. "
            "Không hỏi 'bạn đã thử cách nào chưa' nếu người dùng chưa trực tiếp xin giải pháp. "
            "Hỏi tối đa 1 câu cụ thể để làm rõ điểm then chốt tiếp theo, hoặc đưa 1 bước rất nhỏ nếu dữ kiện đã đủ rõ."
        )

    return (
        "Lượt này chủ yếu cần làm rõ bối cảnh. Phản hồi thấu cảm ngắn và hỏi tối đa 2 câu cụ thể về thông tin còn thiếu."
    )


def build_prompt(
    query: str,
    context: Dict[str, Any],
    history: Optional[List[Dict]],
) -> tuple:
    """
    Build dynamic prompt based on severity, retrieved context, and dialogue state.

    Returns:
        (prompt, system_prompt, severity_level)
    """
    severity = context["severity_indicators"]["level"]
    context_str = _format_context(context)
    history_str = _format_history(history)
    known_facts_str, user_turn_count = _format_known_facts(history)
    turn_guidance = _build_turn_guidance(query, severity, user_turn_count)

    if severity >= TRIAGE_THRESHOLD_HIGH:
        system_prompt = SYSTEM_PROMPT_CRISIS
        micro_few_shot = MICRO_FEW_SHOT_CRISIS
    else:
        system_prompt = SYSTEM_PROMPT_COUNSELING
        micro_few_shot = MICRO_FEW_SHOT_COUNSELING

    prompt_parts = [
        "\n=== YÊU CẦU ĐỊNH DẠNG ===",
        (
            "Trả lời trực tiếp bằng tiếng Việt tự nhiên. "
            "Xưng 'tôi' và gọi người dùng là 'bạn', không gọi là 'em' nếu người dùng chưa tự xưng như vậy. "
            "Không in tiêu đề, nhãn quy trình, checklist, ví dụ mẫu, hoặc tên các bước tư vấn. "
            "Không suy diễn thêm tình huống cụ thể nếu người dùng chưa nói rõ; giữ đúng phạm vi họ mô tả và hỏi làm rõ khi có thể hiểu theo nhiều cách. "
            "Không đặt câu hỏi dẫn dắt có giả định người dùng đã làm điều chưa từng nói. "
            "Không hỏi lại thông tin đã có trong lịch sử hội thoại hoặc dữ kiện người dùng đã cung cấp. "
            "Không biến dữ kiện người dùng vừa nói thành câu hỏi xác nhận; nếu người dùng đã nói thời gian, tần suất, thời điểm, cách đã thử hoặc phần khó nhất thì phải dùng dữ kiện đó để đi tiếp. "
            "Nếu đây là lượt đầu và chưa đủ thông tin, phản hồi ngắn gọn trong 1 đoạn và hỏi tối đa 2 câu cụ thể. "
            "Chỉ đưa giải pháp khi đã có đủ bối cảnh từ hội thoại hoặc khi người dùng đã nêu rõ ràng buộc hành động cụ thể."
        ),
        "\n=== HƯỚNG XỬ LÝ CHO LƯỢT NÀY ===",
        turn_guidance,
        "\n=== MẪU PHONG CÁCH NGẮN ===",
        micro_few_shot,
        "\n=== NGỮ CẢNH RAG ===",
        context_str,
        "\n=== LỊCH SỬ HỘI THOẠI GẦN ĐÂY ===",
        history_str,
        "\n=== DỮ KIỆN NGƯỜI DÙNG ĐÃ CUNG CẤP ===",
        known_facts_str,
        "\n=== CÂU HỎI HIỆN TẠI CỦA NGƯỜI DÙNG ===",
        query,
        "\n=== PHẢN HỒI TỰ NHIÊN CHO NGƯỜI DÙNG ==="
    ]

    prompt = "\n".join(prompt_parts)

    logger.info(f"Built prompt with severity={severity}, using {'CRISIS' if severity>=4 else 'COUNSELING'} mode")

    return prompt, system_prompt, severity
