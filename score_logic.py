import torch
import uuid
from datetime import datetime, timezone
from bson import ObjectId
from sentence_transformers import util

WEIGHTS = {
    "industry": 20,
    "employee_count": 15,
    "location": 15,
    "tech_stack": 25,
    "keywords": 10,
    "founded_year": 5,
    "github_signal": 10
}

def tokenize_icp(icp):
    """
    Tokenizes the ICP filters into a dictionary of sets for efficient lookup,
    and a flat list of tokens for printing.
    """
    icp_filters = icp.get('filters', {})
    icp_field_tokens = {}
    all_tokens = []

    # Extract industry tokens
    industry_tokens = []
    for field_name in ['industry', 'industries']:
        if field_name in icp_filters:
            industries = icp_filters[field_name]
            if isinstance(industries, list):
                for industry in industries:
                    industry_tokens.extend(str(industry).lower().split())
            elif isinstance(industries, str):
                industry_tokens.extend(industries.lower().split())
    if industry_tokens:
        icp_field_tokens['industry'] = set(industry_tokens)
        all_tokens.extend(industry_tokens)

    # Extract tech stack tokens
    tech_tokens = []
    if 'tech_stack' in icp_filters:
        tech_stack = icp_filters['tech_stack']
        if isinstance(tech_stack, list):
            for tech in tech_stack:
                tech_tokens.append(str(tech).lower())
        elif isinstance(tech_stack, str):
            tech_tokens.append(tech_stack.lower())
    if tech_tokens:
        icp_field_tokens['tech_stack'] = set(tech_tokens)
        all_tokens.extend(tech_tokens)

    # Extract keyword tokens
    keyword_tokens = []
    if 'keywords' in icp_filters:
        keywords = icp_filters['keywords']
        if isinstance(keywords, list):
            for keyword in keywords:
                keyword_tokens.append(str(keyword).lower())
        elif isinstance(keywords, str):
            keyword_tokens.append(keywords.lower())
    if keyword_tokens:
        icp_field_tokens['keywords'] = set(keyword_tokens)
        all_tokens.extend(keyword_tokens)

    # Extract location tokens
    location_tokens = []
    for field_name in ['location', 'locations']:
        if field_name in icp_filters:
            locations = icp_filters[field_name]
            if isinstance(locations, list):
                for location in locations:
                    location_tokens.extend(str(location).lower().split())
            elif isinstance(locations, str):
                location_tokens.extend(locations.lower().split())
    if location_tokens:
        icp_field_tokens['location'] = set(location_tokens)
        all_tokens.extend(location_tokens)

    # Employee count and founded after are not tokenized, but used directly in rule_score
    if 'employee_count' in icp_filters:
        icp_field_tokens['employee_count'] = icp_filters['employee_count']
    if 'founded_after' in icp_filters:
        icp_field_tokens['founded_after'] = icp_filters['founded_after']

    return list(set(all_tokens)), icp_field_tokens

def rule_score(company_doc, icp_tokens, icp_field_tokens):
    rule_score_val = 0
    breakdown = {k: 0 for k in WEIGHTS}

    def safe_list(field):
        value = company_doc.get(field, [])
        if isinstance(value, str): return [value]
        if isinstance(value, list): return value
        return []

    # 1. Industry Matching (Whole Word)
    if 'industry' in icp_field_tokens:
        company_industries_text = " ".join(safe_list("industries")).lower()
        company_industry_words = set(company_industries_text.split())
        if not company_industry_words.isdisjoint(icp_field_tokens['industry']):
            rule_score_val += WEIGHTS["industry"]
            breakdown["industry"] = WEIGHTS["industry"]

    # 2. Location Matching (Whole Word)
    if 'location' in icp_field_tokens:
        company_location_parts = []
        hq_location = company_doc.get("hq_location")
        if isinstance(hq_location, str):
            company_location_parts.append(hq_location)
        elif isinstance(hq_location, list):
            company_location_parts.extend(hq_location)
        elif isinstance(hq_location, dict):
            company_location_parts.extend(hq_location.values())

        company_location_text = " ".join(str(p) for p in company_location_parts).lower()
        company_location_words = set(company_location_text.split())
        if not company_location_words.isdisjoint(icp_field_tokens['location']):
            rule_score_val += WEIGHTS["location"]
            breakdown["location"] = WEIGHTS["location"]

    # 3. Tech Stack Matching (Exact Match)
    if 'tech_stack' in icp_field_tokens:
        company_tech = {v.lower() for v in safe_list("tech_stack")}
        if not company_tech.isdisjoint(icp_field_tokens['tech_stack']):
            rule_score_val += WEIGHTS["tech_stack"]
            breakdown["tech_stack"] = WEIGHTS["tech_stack"]

    # 4. Keyword Matching
    if 'keywords' in icp_field_tokens:
        text_to_search = " ".join([
            company_doc.get("name", ""),
            " ".join(safe_list("industries")),
            " ".join(safe_list("tech_stack"))
        ]).lower()
        if any(token in text_to_search for token in icp_field_tokens['keywords']):
            rule_score_val += WEIGHTS["keywords"]
            breakdown["keywords"] = WEIGHTS["keywords"]

    # 5. Employee Count Matching
    if 'employee_count' in icp_field_tokens:
        emp = company_doc.get("employee_count_estimate", {})
        emp_min = emp.get("min", 0)
        emp_max = emp.get("max", float('inf'))
        icp_emp = icp_field_tokens['employee_count']
        icp_min = icp_emp.get('min')
        icp_max = icp_emp.get('max')
        if icp_min is not None and icp_max is not None:
            if not (emp_max < icp_min or emp_min > icp_max):
                rule_score_val += WEIGHTS["employee_count"]
                breakdown["employee_count"] = WEIGHTS["employee_count"]

    # 6. Founded Year Matching
    if 'founded_after' in icp_field_tokens:
        founded_year = company_doc.get("founded_year")
        if isinstance(founded_year, int) and founded_year >= icp_field_tokens['founded_after']:
            rule_score_val += WEIGHTS["founded_year"]
            breakdown["founded_year"] = WEIGHTS["founded_year"]

    # 7. GitHub Signal
    urls = company_doc.get("source_urls", [])
    if any("github.com" in url.lower() for url in urls):
        rule_score_val += WEIGHTS["github_signal"]
        breakdown["github_signal"] = WEIGHTS["github_signal"]

    return rule_score_val, breakdown

def score_companies(search_result, icp, icp_embedding, discovered_companies, scored_companies):
    icp_id = str(icp["_id"])
    icp_version = icp.get("version", 1)
    icp_tokens, icp_field_tokens = tokenize_icp(icp)

    print(f"✅ Extracted ICP tokens: {icp_tokens}")
    print(f"✅ Field-specific tokens: {icp_field_tokens}")

    scored_results_util = []
    icp_tensor = torch.tensor([icp_embedding], dtype=torch.float32)

    for point in search_result.points:
        payload = point.payload
        company_id = payload.get("company_id")
        company_vector = point.vector

        company_tensor = torch.tensor([company_vector], dtype=torch.float32)
        raw_similarity = util.cos_sim(icp_tensor, company_tensor)[0][0].item()
        vector_similarity_score = ((raw_similarity + 1) / 2) * 100

        try:
            if isinstance(company_id, str):
                company_id = ObjectId(company_id)
        except Exception as e:
            print(f'{company_id} has invalid ObjectId: {e}')
            continue

        company = discovered_companies.find_one({"_id": company_id})
        if not company:
            continue

        rule_score_total, breakdown = rule_score(company, icp_tokens, icp_field_tokens)

        vector_weight = 0.4
        rule_weight = 0.6

        final_score = round(
            (vector_similarity_score * vector_weight) + (rule_score_total * rule_weight),
            2
        )

        breakdown["vector_similarity"] = round(vector_similarity_score, 2)

        scored_doc = {
            "company_id": company_id,
            "icp_id": icp_id,
            "icp_version": icp_version,
            "final_score": final_score,
            "breakdown": breakdown,
            "weights": WEIGHTS,
            "last_scored": datetime.now(timezone.utc),
            "method": "util_cos_sim"
        }

        scored_companies.update_one(
            {"company_id": company_id, "icp_id": icp_id, "method": "util_cos_sim"},
            {"$set": scored_doc},
            upsert=True
        )

        scored_results_util.append((company.get("domain", "unknown"), final_score))

    return scored_results_util
