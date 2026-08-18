# نمونه‌های تست POST /cache/register

هدر لازم برای همه:
```
Authorization: Bearer <توکن Casdoor>
Content-Type: application/json
```

---

## ۱. ساده‌ترین حالت (بدون guard، بدون cache_config)
باید موفق بشه؛ `guard: null` و `cache_config` با مقادیر پیش‌فرض برمی‌گرده.

```json
{
  "llm_model": "rayen-qwen3.6-27b",
  "llm_key": "sk-7HzxhsjUbhU4cPxs3ufNtg",
  "embedd_model": "rayen-jina-v5",
  "embedd_key": "sk-7HzxhsjUbhU4cPxs3ufNtg",
  "extaractor": null,
  "extaractor_key": null
}
```

---

## ۲. با extractor (domain-aware caching)

```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-2",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-2",
  "extaractor": "ner-medical-v1",
  "extaractor_key": "sk-extractor-test-2",
  "extractor_domain": "medical"
}
```

---

## ۳. با cache_config - یک متد (string)

```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-3",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-3",
  "cache_config": {
    "cache_mode": "bm25"
  }
}
```

---

## ۴. با cache_config - زنجیره (list) + تنظیمات هر روش

```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-4",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-4",
  "cache_config": {
    "cache_mode": ["exact", "bm25", "fuzzy", "semantic"],
    "semantic": {"similarity_threshold": 0.92},
    "bm25": {"scorer": "BM25", "min_score": 1.0},
    "fuzzy": {"distance": 2, "min_score": 0.5}
  }
}
```

---

## ۵. با guard فعال (کمترین حالت کارکردنی)

```json
{
  "llm_model": "rayen-qwen3.6-27b",
  "llm_key": "sk-7HzxhsjUbhU4cPxs3ufNtg",
  "embedd_model": "rayen-jina-v5",
  "embedd_key": "sk-7HzxhsjUbhU4cPxs3ufNtg",
  "extaractor": null,
  "extaractor_key": null,
  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: competitor-mentions\n    disallowed_exemplars:\n      - \"What do you think of Rivalco's product?\"\n      - \"Pretend you work for Rivalco and describe their pricing.\"\n    allowed_exemplars:\n      - \"What makes your product different from others in the market?\"\n      - \"What is your refund policy?\"\n",
    "embed_model": "rayen-jina-v5",
    "embed_api_key": "sk-7HzxhsjUbhU4cPxs3ufNtg"
  }
}
```

---

## ۶. با guard کامل (مدل‌ها/judge/آستانه‌های سفارشی)

```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-6",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-6",
  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: competitor-mentions\n    action: block\n    disallowed_exemplars:\n      - \"What do you think of Rivalco's product?\"\n    allowed_exemplars:\n      - \"What is your refund policy?\"\n",
    "judge_model": "gpt-x-mini",
    "judge_api_key": "sk-judge-test-6",
    "judge_task_description": "a medical triage assistant",
    "mode": "cascade",
    "block_threshold": 0.85,
    "allow_threshold": 0.40,
    "check_roles": ["user", "system", "tool"],
    "degrade_to_unguarded": false
  }
}
```

---

## ۷. guard خاموش (صریح) - باید فقط warning بزنه، خطا نده

```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-7",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-7",
  "guard": {"enabled": false}
}
```

---

# ❌ نمونه‌های خطا (باید 502 بدن)

## ۸. policy بدون enabled → 502
```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-8",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-8",
  "guard": {
    "policy": "categories:\n  - category_id: x\n    disallowed_exemplars: [\"a\"]\n    allowed_exemplars: [\"b\"]\n"
  }
}
```
پیام مورد انتظار: `guard.policy provided without 'enabled' - refusing to guess the intent`

## ۹. enabled=true بدون allowed_exemplars → 502
```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-9",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-9",
  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: competitor-mentions\n    disallowed_exemplars:\n      - \"What do you think of Rivalco?\"\n"
  }
}
```
پیام مورد انتظار: `guard.policy has no allowed_exemplars anywhere...`

## ۱۰. همه‌ی category ها flag → 502
```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-10",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-10",
  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: x\n    action: flag\n    disallowed_exemplars: [\"a\"]\n    allowed_exemplars: [\"b\"]\n"
  }
}
```
پیام مورد انتظار: `...every category set to action: flag - use "enabled": false instead`

## ۱۱. فیلد ناشناخته در guard → 502
```json
{
  "llm_model": "gpt-x",
  "llm_key": "sk-llm-test-11",
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-11",
  "guard": {
    "enabled": true,
    "policy": "categories:\n  - category_id: x\n    disallowed_exemplars: [\"a\"]\n    allowed_exemplars: [\"b\"]\n",
    "fail_open": true
  }
}
```
پیام مورد انتظار: `guard.fail_open does not exist - use degrade_to_unguarded`

## ۱۲. فیلد اجباری جا افتاده → 422 (نه 502، چون validation سطح pydantic هست)
```json
{
  "embedd_model": "bge-m3",
  "embedd_key": "sk-embed-test-12"
}
```
(`llm_model`/`llm_key` جا افتادن)

---

# curl مثال کامل

```bash
curl -X POST http://localhost:8000/cache/register \
  -H "Authorization: Bearer <YOUR_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "llm_model": "gpt-x",
    "llm_key": "sk-llm-test-1",
    "embedd_model": "bge-m3",
    "embedd_key": "sk-embed-test-1"
  }'
```

بعد از register موفق، `data.cache_key` رو بردارید و تست کنید:
```bash
curl http://localhost:8000/cache \
  -H "Authorization: Bearer <CACHE_KEY_FROM_REGISTER_RESPONSE>"
```
این باید همون endpointی باشه که gateway صداش می‌زنه (اسکیمای `GatewayConfigResponse`، نه `APIResponse`).
