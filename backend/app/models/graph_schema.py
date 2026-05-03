"""
Graph Schema Definition for Neo4j Knowledge Graph
Defines node labels, relationship types, and properties
"""
from typing import Dict, List, Any

# Node Labels
NODE_LABELS = {
    "BenhLy": {
        "description": "Psychological disorders/mental health conditions",
        "properties": {
            "id": "Unique identifier (UUID)",
            "name": "Standard medical name",
            "aliases": "List of alternative names/symptoms",
            "severity_level": "Integer 1-5, higher = more severe",
            "source_domain": "Source document category",
            "icd_code": "Optional ICD-10 code",
            "description": "Brief description",
            "treatment_approach": "General treatment approach"
        }
    },
    "TrieuChung": {
        "description": "Symptoms/signs",
        "properties": {
            "id": "Unique identifier",
            "name": "Symptom name",
            "aliases": "Alternative expressions",
            "category": "Emotional, Cognitive, Behavioral, Physical",
            "severity_level": "Integer 1-5",
            "is_red_flag": "Boolean - indicates danger signal"
        }
    },
    "DauHieuNguyHiem": {
        "description": "Red flags / crisis indicators",
        "properties": {
            "id": "Unique identifier",
            "name": "Danger sign name",
            "description": "Detailed explanation",
            "action_required": "Immediate action needed",
            "severity_level": "Integer 4-5 (always high)"
        }
    },
    "HanhDongPFA": {
        "description": "Psychological First Aid actions",
        "properties": {
            "id": "Unique identifier",
            "name": "Action name",
            "category": "Look, Listen, or Link",
            "description": "How to perform this action",
            "target_group": "Applicable demographic groups"
        }
    },
    "KyNangTuVan": {
        "description": "Counseling micro-skills",
        "properties": {
            "id": "Unique identifier",
            "name": "Skill name",
            "category": "Listening, Questioning, Responding, etc.",
            "description": "How to apply this skill",
            "age_appropriate": "Age groups this skill works best for"
        }
    },
    "BuocTuVan": {
        "description": "Counseling process steps",
        "properties": {
            "id": "Unique identifier",
            "name": "Step name",
            "order": "Integer indicating sequence",
            "description": "What happens in this step",
            "key_skills": "Skills used in this step"
        }
    },
    "Thuoc": {
        "description": "Medications (for identification only, NOT prescription)",
        "properties": {
            "id": "Unique identifier",
            "name": "Drug name",
            "category": "Antidepressant, Antipsychotic, Anxiolytic, etc.",
            "description": "Brief description",
            "warning": "Important warnings (e.g., MUST be prescribed)"
        }
    },
    "DoiTuong": {
        "description": "Demographic groups",
        "properties": {
            "id": "Unique identifier",
            "name": "Group name",
            "age_range": "e.g., 6-11, 11-15, 16-18",
            "vulnerability_factors": "Specific vulnerabilities",
            "communication_style": "Recommended approach"
        }
    },
    "DocumentChunk": {
        "description": "Chunk of source document",
        "properties": {
            "id": "Unique identifier",
            "text": "Chunk text content",
            "source": "Source file path",
            "section": "Document section",
            "doc_type": "medical_guideline, first_aid, school_counseling",
            "risk_priority": "high/medium/low",
            "page_no": "Page number",
            "embedding": "Vector embedding (optional)"
        }
    }
}

# Relationship Types
RELATIONSHIPS = {
    "CO_TRIEU_CHUNG": {
        "description": "Disorder has symptom",
        "from": "BenhLy",
        "to": "TrieuChung",
        "properties": {
            "weight": "Confidence/strength of association (0-1)"
        }
    },
    "BAO_HIEU_NGUY_HIEM": {
        "description": "Symptom/disorder indicates danger",
        "from": ["TrieuChung", "BenhLy"],
        "to": "DauHieuNguyHiem",
        "properties": {
            "weight": "Association strength"
        }
    },
    "YEU_CAU_HANH_DONG": {
        "description": "Danger sign requires specific action",
        "from": ["DauHieuNguyHiem", "BenhLy"],
        "to": "HanhDongPFA",
        "properties": {
            "priority": "Action priority level"
        }
    },
    "DIEU_TRI_BANG": {
        "description": "Disorder is treated with medication",
        "from": "BenhLy",
        "to": "Thuoc",
        "properties": {
            "is_first_line": "Boolean - first line treatment",
            "dosage_info": "Optional dosage info for ID purposes only"
        }
    },
    "QUAN_LY_BANG": {
        "description": "Issue managed with counseling skill",
        "from": ["TrieuChung", "BenhLy"],
        "to": ["KyNangTuVan", "BuocTuVan"],
        "properties": {
            "effectiveness": "How effective (1-5)"
        }
    },
    "AP_DUNG_CHO": {
        "description": "Skill/action applies to demographic",
        "from": ["KyNangTuVan", "HanhDongPFA"],
        "to": "DoiTuong",
        "properties": {
            "adaptation_needed": "Boolean - needs age adaptation"
        }
    },
    "BAO_GOM_BUOC": {
        "description": "Step contains substep",
        "from": "BuocTuVan",
        "to": "BuocTuVan",
        "properties": {
            "order": "Sequence within parent step"
        }
    },
    "NAM_TRONG_CHUNK": {
        "description": "Entity mentioned in document chunk",
        "from": ["BenhLy", "TrieuChung", "Thuoc", "KyNangTuVan", "HanhDongPFA"],
        "to": "DocumentChunk",
        "properties": {
            "relevance_score": "How relevant (0-1)"
        }
    }
}

# Predefined entities for initialization (from the design document)
PREDEFINED_ENTITIES = {
    "BenhLy": [
        {"name": "Bệnh Alzheimer", "aliases": ["sa sút trí tuệ", "alzheimer", "dement"], "severity_level": 4},
        {"name": "Tâm thần phân liệt", "aliases": ["schizophrenia", "ảo thanh", "hoang tưởng"], "severity_level": 5},
        {"name": "Trầm cảm", "aliases": ["buồn bã", "tuyệt vọng", "depression"], "severity_level": 3},
        {"name": "Rối loạn lo âu", "aliases": ["lo lắng", "căng thẳng", "anxiety"], "severity_level": 3},
        {"name": "Rối loạn Tăng động Giảm chú ý", "aliases": ["ADHD", "tăng động"], "severity_level": 2},
        {"name": "Sa sút trí tuệ bệnh mạch máu", "aliases": ["vascular dementia"], "severity_level": 4},
        {"name": "Rối loạn nhân cách", "aliases": ["borderline", "nhân cách"], "severity_level": 4}
    ],
    "DauHieuNguyHiem": [
        {"name": "Ý định tự sát", "description": "Có ý định tự làm hại bản thân", "severity_level": 5},
        {"name": "Hành vi tự hại", "description": "Tự cắt, đánh, bỏ ăn...", "severity_level": 5},
        {"name": "Ảo giác/Ảo thanh", "description": "Nghe thấy tiếng nói, thấy hình ảnh không tồn tại", "severity_level": 4},
        {"name": "Hoang tưởng", "description": "Tin tưởng sai lệch, khó thay đổi", "severity_level": 4},
        {"name": "Bạo lực", "description": "Hành vi hung hăng, đe dọa làm hại người khác", "severity_level": 5},
        {"name": "Bị bạo lực/xâm hại", "description": "Là nạn nhân của bạo lực thể chất/tình dục", "severity_level": 5}
    ],
    "HanhDongPFA": [
        {"name": "Quan sát an toàn", "category": "Look", "description": "Đánh giá môi trường an toàn"},
        {"name": "Lắng nghe nhu cầu khẩn cấp", "category": "Listen", "description": "Lắng nghe mà không phán xét"},
        {"name": "Kỹ thuật giúp bình tĩnh", "category": "Listen", "description": "Hướng dẫn hít thở, grounding"},
        {"name": "Chuyển tuyến y tế", "category": "Link", "description": "Hướng dẫn tìm dịch vụ chuyên nghiệp"},
        {"name": "Kết nối với người tin cậy", "category": "Link", "description": "Khuyến khích chia sẻ với người thân"}
    ],
    "KyNangTuVan": [
        {"name": "Lắng nghe tích cực", "category": "Listening"},
        {"name": "Phản hồi cảm xúc", "category": "Responding"},
        {"name": "Đặt câu hỏi mở", "category": "Questioning"},
        {"name": "Thấu cảm", "category": "Responding"},
        {"name": "Xử lý im lặng", "category": "Process"}
    ],
    "BuocTuVan": [
        {"name": "Xây dựng quan hệ", "order": 1},
        {"name": "Làm rõ vấn đề", "order": 2},
        {"name": "Lựa chọn giải pháp", "order": 3},
        {"name": "Triển khai thực hiện", "order": 4},
        {"name": "Lượng giá kết quả", "order": 5},
        {"name": "Theo dõi sau kết thúc", "order": 6}
    ],
    "DoiTuong": [
        {"name": "Học sinh tiểu học", "age_range": "6-11"},
        {"name": "Học sinh THCS", "age_range": "11-15"},
        {"name": "Học sinh THPT", "age_range": "16-18"},
        {"name": "Trẻ không có người lớn đi cùng", "vulnerability": "High"},
        {"name": "Phụ nữ mang thai", "vulnerability": "Medium"},
        {"name": "Người khuyết tật", "vulnerability": "High"}
    ],
    "Thuoc": [
        {"name": "Sertraline", "category": "Antidepressant", "warning": "Chỉ dùng theo đơn bác sĩ"},
        {"name": "Fluoxetine", "category": "Antidepressant", "warning": "Chỉ dùng theo đơn bác sĩ"},
        {"name": "Haloperidol", "category": "Antipsychotic", "warning": "Chỉ dùng theo đơn bác sĩ"},
        {"name": "Risperidone", "category": "Antipsychotic", "warning": "Chỉ dùng theo đơn bác sĩ"},
        {"name": "Clozapine", "category": "Antipsychotic", "warning": "Chỉ dùng theo đơn bác sĩ"},
        {"name": "Donepezil", "category": "Cognitive enhancer", "warning": "Chỉ dùng theo đơn bác sĩ"},
        {"name": "Valproate", "category": "Mood stabilizer", "warning": "Chỉ dùng theo đơn bác sĩ"}
    ]
}


def get_schema_info() -> Dict[str, Any]:
    """Return complete schema information"""
    return {
        "node_labels": NODE_LABELS,
        "relationships": RELATIONSHIPS,
        "predefined_entities": PREDEFINED_ENTITIES
    }