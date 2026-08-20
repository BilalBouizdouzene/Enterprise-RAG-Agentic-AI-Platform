"""Generate a fictional, reproducible enterprise corpus and evaluation sets."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
EVAL = ROOT / "data" / "evaluation"


DOCUMENTS = [
    ("hr", "hr_remote_work", "Remote Work Policy", "HR", "People", "2025-01-15", """# Purpose
This policy defines hybrid work eligibility for employees of Northstar Analytics.

## Eligibility
Employees may work remotely up to three days per week with manager approval. Customer-facing roles may have different schedules.

## Equipment and security
The company provides an encrypted laptop and requires VPN access on untrusted networks. Lost equipment must be reported to Security within one hour.

## Review
The policy is reviewed annually by Human Resources and Security.
"""),
    ("hr", "hr_leave", "Leave and Time Off Policy", "HR", "People", "2024-06-01", """# Annual leave
Full-time employees receive 25 paid working days per calendar year. Requests should be submitted ten business days in advance.

## Sick leave
Employees must notify their manager before the start of the working day. Local statutory requirements remain applicable.

## Parental leave
Parental leave is administered according to the employee's country of employment and applicable law.
"""),
    ("hr", "hr_onboarding", "Employee Onboarding Guide", "HR", "People", "2025-02-10", """# First week
New employees complete security awareness training, receive an identity account, and meet their department lead.

## Required access
The manager requests access through the service desk. Least privilege applies from the first day.

## France and Germany
The People team schedules a local orientation for employees based in France or Germany.
"""),
    ("finance", "finance_expenses", "Expense Reimbursement Policy", "Finance", "Expenses", "2025-03-01", """# Eligible expenses
Reasonable business travel expenses are reimbursable when supported by a receipt and an approved trip.

## Accommodation limits
The maximum hotel reimbursement is EUR 180 per night in Paris, EUR 150 in other French cities, and EUR 160 in Berlin.

## Meals and submission
Meals are capped at EUR 45 per day. Employees submit expenses within 30 days.
"""),
    ("finance", "finance_approvals", "Purchase Approval Matrix", "Finance", "Procurement", "2024-09-12", """# Approval levels
Purchases up to EUR 500 require the budget owner. Purchases from EUR 501 to EUR 5,000 require the department director.

## Higher value purchases
Purchases above EUR 5,000 require Finance review and a procurement record. Security review is required for software handling confidential data.
"""),
    ("finance", "finance_travel", "Business Travel Standard", "Finance", "Travel", "2025-01-20", """# Travel booking
Employees should use the approved travel portal and select economy rail or air for journeys under six hours.

## Limits
Hotel limits are EUR 180 in Paris, EUR 160 in Berlin, and EUR 140 in other European cities. A director may approve an exception before booking.

## Reconciliation
Travel expenses must include the business purpose, project code, and receipt.
"""),
    ("security", "security_laptop", "Lost Laptop Procedure", "Security", "Incident Response", "2025-04-05", """# Immediate response
Report a lost laptop to the Security Operations mailbox and service desk within one hour. Do not attempt to recover it alone.

## Containment
Security remotely locks the device, revokes active sessions, and assesses whether company data may be exposed.

## Follow-up
The employee provides the location, time, device asset tag, and last known network. Security records the incident severity.
"""),
    ("security", "security_passwords", "Password and Access Control Standard", "Security", "Identity", "2025-02-01", """# Passwords
Passwords must be at least 14 characters, unique to Northstar, and protected with a password manager. Passwords must not be shared through email or chat.

## Authentication
Multi-factor authentication is required for cloud applications and administrative access. Dormant accounts are reviewed monthly.

## Access reviews
Managers review department access every quarter. The Security team removes access within one business day after termination.
"""),
    ("legal", "legal_retention", "Data Retention Standard", "Legal", "Compliance", "2024-11-18", """# Retention principle
Northstar retains information only as long as required for a documented business or legal purpose.

## Requests
The Legal team maintains the retention schedule. Employees must not delete records subject to a legal hold.

## Personal data
Personal data is minimized, access-controlled, and securely deleted at the end of its retention period.
"""),
    ("legal", "legal_vendors", "Vendor Compliance Procedure", "Legal", "Third Party Risk", "2025-05-22", """# Due diligence
Vendors processing confidential or personal data require a security questionnaire, data-processing agreement, and risk assessment before onboarding.

## Monitoring
Critical vendors are reviewed annually. Material incidents must be reported to Northstar within 24 hours of discovery.

## Ownership
Procurement owns the vendor record, Security owns technical risk, and Legal approves contractual terms.
"""),
]


EXTRA_DOCUMENTS = [
    ("hr", "hr_benefits", "Employee Benefits Guide", "HR", "Benefits", "2025-02-18", """# Eligibility
Employees receive healthcare, meal support, and a learning allowance according to their country and contract.

## Requests
Benefit requests are submitted through the People portal and reviewed within five business days.

## Changes
Employees must report a change of family status to People Operations so records remain accurate.
"""),
    ("hr", "hr_performance", "Performance Review Framework", "HR", "Performance", "2024-10-03", """# Review cycle
Managers hold a quarterly check-in and complete a documented annual performance review.

## Objectives
Objectives must be measurable, connected to team priorities, and reviewed when business priorities change.

## Development
Employees may request coaching, mentoring, and training through their manager.
"""),
    ("hr", "hr_training", "Learning and Training Standard", "HR", "Learning", "2025-01-08", """# Required training
Security awareness and privacy training are required during onboarding and annually thereafter.

## Records
The Learning team records completion and reminds employees before the due date.

## External courses
External courses require manager approval and a documented business purpose.
"""),
    ("hr", "hr_recruitment", "Recruitment and Interview Guide", "HR", "Recruitment", "2024-08-11", """# Fair recruitment
Hiring decisions use job-related evidence and must not discriminate against candidates.

## Interview records
Interview notes are stored in the approved recruitment system and access is limited to the hiring panel.

## Approval
The hiring manager and People partner approve the final offer before it is sent.
"""),
    ("hr", "hr_payroll", "Payroll Calendar and Change Procedure", "HR", "Payroll", "2025-01-05", """# Payroll calendar
Payroll is processed on the last business day of each month. Local bank holidays may change the payment date.

## Changes
Employees submit bank-account changes through the People portal with identity verification.

## Questions
Payroll questions should be raised through the service desk before the monthly close.
"""),
    ("finance", "finance_cards", "Corporate Card Standard", "Finance", "Expenses", "2025-02-14", """# Card usage
Corporate cards are for approved business purchases and may not be used for personal expenses.

## Reconciliation
Cardholders attach receipts and business purpose within 15 days of the transaction.

## Exceptions
Lost cards must be reported to Finance and the card issuer immediately.
"""),
    ("finance", "finance_invoices", "Invoice Processing Procedure", "Finance", "Accounts Payable", "2024-12-01", """# Submission
Supplier invoices must include a purchase order, legal entity, tax details, and payment terms.

## Review
The budget owner confirms receipt of goods before Finance releases an invoice for payment.

## Timing
Standard invoices are targeted for payment within 30 days after approval.
"""),
    ("finance", "finance_budget", "Budget Planning Handbook", "Finance", "Planning", "2025-04-12", """# Planning cycle
Departments submit an annual operating plan with assumptions, headcount, and expected expenses.

## Variance
Budget owners review monthly variance and explain material changes to Finance.

## Controls
Unplanned commitments require approval before a supplier is engaged.
"""),
    ("finance", "finance_supplier_payments", "Supplier Payment Controls", "Finance", "Procurement", "2024-07-15", """# Payment controls
Supplier bank-account changes require independent verification and dual approval.

## Segregation
The person who creates a supplier must not approve the supplier payment.

## Evidence
Finance retains approval evidence, invoice records, and confirmation of delivery.
"""),
    ("security", "security_vpn", "Remote Network and VPN Standard", "Security", "Network", "2025-03-14", """# VPN use
Employees must use the corporate VPN when connecting from public or untrusted networks.

## Credentials
VPN credentials are personal, protected by multi-factor authentication, and must not be shared.

## Monitoring
Security monitors unusual login locations and may suspend a session for investigation.
"""),
    ("security", "security_data_classification", "Data Classification Policy", "Security", "Data Protection", "2024-05-09", """# Classification levels
Information is classified as Public, Internal, Confidential, or Restricted according to business impact.

## Handling
Confidential data requires access control, approved storage, and encryption in transit.

## Ownership
Data owners review classification annually and after a major process change.
"""),
    ("security", "security_incident_response", "Security Incident Response Plan", "Security", "Incident Response", "2025-06-01", """# Detection
Employees report suspected phishing, unauthorized access, or data exposure to Security immediately.

## Triage
Security assigns severity, preserves evidence, and coordinates containment with IT and Legal.

## Communication
Only the Incident Lead communicates externally during an active investigation.
"""),
    ("security", "security_access_requests", "Application Access Request Procedure", "Security", "Identity", "2024-10-22", """# Request
Managers request application access through the service desk and state the business justification.

## Approval
The application owner approves access and Security checks least privilege for sensitive systems.

## Review
Access is reviewed quarterly and removed when the business need ends.
"""),
    ("security", "security_phishing", "Phishing Awareness Playbook", "Security", "Awareness", "2025-02-26", """# Suspicious messages
Do not click links or open attachments in a suspicious message. Use the report-phishing button instead.

## Credentials
If credentials were entered, change the password and contact Security immediately.

## Follow-up
Security analyzes message indicators and shares lessons learned without blaming the reporter.
"""),
    ("legal", "legal_privacy_requests", "Privacy Request Procedure", "Legal", "Privacy", "2025-01-30", """# Intake
Privacy requests are logged by the Privacy team and identity is verified before disclosure.

## Response
The team coordinates with HR, IT, and business owners to locate relevant personal data.

## Deadline
Requests are handled within the applicable legal deadline and all decisions are documented.
"""),
    ("legal", "legal_contracts", "Contract Review Standard", "Legal", "Contracts", "2024-09-20", """# Review
Material customer and supplier contracts require Legal review before signature.

## Risk
The reviewer checks liability, confidentiality, data processing, termination, and governing law.

## Records
Signed contracts are stored in the contract repository with an owner and renewal date.
"""),
    ("legal", "legal_legal_hold", "Legal Hold Procedure", "Legal", "Compliance", "2025-03-07", """# Trigger
Legal issues a hold when litigation or an investigation is reasonably anticipated.

## Preservation
Employees must preserve relevant email, files, messages, and records until Legal releases the hold.

## Ownership
Legal owns the hold notice and documents acknowledgements and exceptions.
"""),
    ("legal", "legal_whistleblowing", "Speak-Up and Investigation Policy", "Legal", "Ethics", "2024-04-17", """# Reporting
Employees may report suspected misconduct through the confidential ethics channel or their manager.

## Protection
Retaliation against a person who raises a good-faith concern is prohibited.

## Investigation
Legal assigns an investigator, protects confidentiality, and records the outcome.
"""),
    ("technical", "it_service_desk", "IT Service Desk Standard", "IT", "Service Management", "2025-01-12", """# Requests
Employees use the service desk for incidents, access requests, equipment, and software support.

## Priority
Critical incidents affecting many users receive priority over individual convenience requests.

## Updates
The service desk records the owner, status, impact, and resolution for every ticket.
"""),
    ("technical", "it_backup", "Backup and Restore Standard", "IT", "Resilience", "2024-11-02", """# Backup
Critical systems are backed up according to their recovery requirements and monitored for failures.

## Restore tests
IT performs restore tests quarterly and records recovery time and data completeness.

## Access
Backup copies are access-controlled and protected from unauthorized deletion.
"""),
    ("technical", "it_change_management", "Change Management Procedure", "IT", "Operations", "2025-05-03", """# Change request
Production changes require a ticket, risk assessment, implementation plan, and rollback plan.

## Approval
The service owner approves normal changes; emergency changes are reviewed after implementation.

## Evidence
The change record includes testing evidence, deployment time, and observed impact.
"""),
    ("technical", "it_api_security", "API Security Standard", "IT", "Application Security", "2025-04-19", """# Authentication
Internal APIs use approved authentication, authorization, and secret-management mechanisms.

## Logging
API access logs exclude secrets and personal data while retaining enough context for investigation.

## Testing
Teams perform dependency scanning and security testing before a material release.
"""),
    ("operations", "operations_business_continuity", "Business Continuity Plan", "Operations", "Resilience", "2025-02-05", """# Critical processes
Business owners identify critical processes, dependencies, recovery objectives, and delegated contacts.

## Exercises
Teams run a continuity exercise at least annually and track corrective actions.

## Incident mode
During a disruption, the Incident Lead coordinates updates and prioritizes employee safety.
"""),
    ("operations", "operations_office_access", "Office Access and Visitor Policy", "Operations", "Facilities", "2024-08-04", """# Badges
Employees display their badge and do not allow unknown people to follow them through secure doors.

## Visitors
Visitors register at reception, wear a visitor badge, and remain accompanied in restricted areas.

## Lost badge
Lost badges are reported to Facilities immediately so access can be disabled.
"""),
    ("operations", "operations_travel_safety", "Travel Safety Guidance", "Operations", "Travel", "2025-03-28", """# Before travel
Employees review destination guidance, emergency contacts, and health requirements before an international trip.

## During travel
Use approved transport and report a safety concern to the travel desk or local emergency services.

## Support
Operations maintains an emergency contact list for active business travel.
"""),
    ("operations", "operations_procurement", "Operational Procurement Guide", "Operations", "Procurement", "2024-10-14", """# Supplier selection
Operational purchases use approved suppliers, documented requirements, and a fair comparison of offers.

## Approval
The budget owner confirms need and Finance checks the purchase approval matrix.

## Records
Procurement retains quotes, approvals, purchase orders, and delivery confirmation.
"""),
]


GENERATED_TOPICS = [
    ("hr", "HR", "People Operations", "People", "The People Operations owner reviews employee records monthly and documents material changes in the approved HR system."),
    ("finance", "Finance", "Financial Control", "Controls", "The Finance owner reconciles the control evidence monthly and escalates a material variance to the budget owner."),
    ("security", "Security", "Security Control", "Information Security", "The Security owner reviews the control quarterly, records exceptions, and confirms that corrective actions have an accountable owner."),
    ("legal", "Legal", "Compliance Review", "Compliance", "The Legal owner maintains the review record, confirms the applicable obligation, and stores evidence in the approved repository."),
    ("technical", "IT", "Technology Operations", "Technology", "The IT owner monitors the service, records operational evidence, and follows the approved change and incident process."),
    ("operations", "Operations", "Business Operations", "Operations", "The Operations owner reviews the operating checklist, records dependencies, and escalates unresolved risks through the service desk."),
]

GENERATED_REGIONS = ["France", "Germany", "Spain", "Belgium", "Netherlands", "United Kingdom", "Italy", "Portugal"]


def generated_documents(count: int) -> list[tuple[str, str, str, str, str, str, str]]:
    """Create a larger, deterministic corpus without pretending it is real company data."""
    records = []
    for index in range(1, count + 1):
        department, department_name, title_prefix, category, control = GENERATED_TOPICS[(index - 1) % len(GENERATED_TOPICS)]
        region = GENERATED_REGIONS[(index - 1) % len(GENERATED_REGIONS)]
        document_id = f"generated_{index:04d}"
        title = f"{title_prefix} Brief {index:04d}"
        body = f"""# Purpose
This synthetic {department_name.lower()} brief covers the {region} operating context for control reference {index:04d}. It is included to exercise enterprise-scale ingestion, filtering, and retrieval locally.

## Operating rule
{control} The local owner confirms the {region} status at least every 30 days and links the decision to reference {index:04d}.

## Evidence
The record includes the responsible team, review date, exception notes, and follow-up action. Questions about this brief should be directed to the {department_name} service owner.
"""
        records.append((department, document_id, title, department_name, category, f"2025-{(index % 12) + 1:02d}-01", body))
    return records


def write_documents(generated_count: int = 600) -> list[dict[str, str]]:
    records = []
    scalable_documents = generated_documents(generated_count)
    corpus = DOCUMENTS + EXTRA_DOCUMENTS + scalable_documents
    generated_ids = {item[1] for item in scalable_documents}
    for department, document_id, title, department_name, category, effective_date, body in corpus:
        folder = RAW / ("generated" if document_id in generated_ids else department)
        folder.mkdir(parents=True, exist_ok=True)
        metadata = (
            "---\n"
            f"document_id: {document_id}\n"
            f"title: {title}\n"
            f"department: {department_name}\n"
            f"category: {category}\n"
            "version: 1.0\n"
            f"effective_date: {effective_date}\n"
            "language: en\n"
            "source: synthetic-demo\n"
            "---\n\n"
        )
        (folder / f"{document_id}.md").write_text(metadata + body.strip() + "\n", encoding="utf-8")
        records.append({"document_id": document_id, "title": title, "department": department_name})
    return records


def write_evaluation(records: list[dict[str, str]]) -> None:
    questions = [
        ("What is the maximum hotel reimbursement in Paris?", ["finance_expenses", "finance_travel"], "180 EUR", "lexical_exact"),
        ("How much can a hotel in Berlin cost under the travel standard?", ["finance_travel"], "160 EUR", "semantic_paraphrase"),
        ("How many paid working days of annual leave are provided?", ["hr_leave"], "25 days", "lexical_exact"),
        ("How often may an eligible employee work remotely?", ["hr_remote_work"], "up to three days per week", "lexical_exact"),
        ("What should happen after a company laptop is lost?", ["security_laptop"], "report within one hour", "lexical_exact"),
        ("What is the minimum password length?", ["security_passwords"], "14 characters", "lexical_exact"),
        ("When must a critical vendor report a material incident?", ["legal_vendors"], "within 24 hours", "lexical_exact"),
        ("Who approves contractual vendor terms?", ["legal_vendors"], "Legal", "lexical_exact"),
        ("What is the approval requirement for a purchase of EUR 3000?", ["finance_approvals"], "department director", "lexical_exact"),
        ("What happens to access after termination?", ["security_passwords"], "removed within one business day", "lexical_exact"),
        ("Which policy became effective in 2025?", ["hr_remote_work", "hr_onboarding", "finance_expenses", "finance_travel", "security_laptop", "security_passwords", "legal_vendors"], "one or more policies", "metadata"),
        ("Compare hotel limits in Paris and Berlin.", ["finance_expenses", "finance_travel"], "Paris 180 EUR and Berlin 160 EUR", "multi_document"),
        ("Compare laptop incident reporting with vendor incident reporting.", ["security_laptop", "legal_vendors"], "one hour versus 24 hours", "multi_document"),
        ("What is the retention period for every customer record?", [], "not specified", "unanswerable"),
        ("What is the maximum meal reimbursement in Madrid?", [], "not specified", "unanswerable"),
    ]
    # Repeat only with deterministic wording variations to create a scalable, labeled set.
    variations = [
        ("Please explain: ", ""),
        ("In simple terms, ", ""),
        ("According to the policies, ", ""),
        ("Summarize this requirement: ", ""),
    ]
    rows = []
    index = 1
    for question, docs, answer, qtype in questions:
        for prefix, suffix in variations:
            rows.append({
                "id": f"q{index:03d}",
                "question": prefix + question[0].lower() + question[1:] + suffix,
                "relevant_document_ids": docs,
                "relevant_chunk_ids": [],
                "expected_answer": answer,
                "question_type": qtype,
                "category": qtype,
            })
            index += 1
    additional_questions = [
        ("What nightly lodging amount can an employee claim in the French capital?", ["finance_expenses", "finance_travel"], "180 EUR", "semantic_paraphrase"),
        ("During a trip to Germany's capital, what accommodation ceiling applies?", ["finance_travel"], "160 EUR", "semantic_paraphrase"),
        ("How many annual paid workdays are granted to a full-time employee each year?", ["hr_leave"], "25 days", "semantic_paraphrase"),
        ("What is the weekly remote-work allowance for an eligible staff member?", ["hr_remote_work"], "up to three days per week", "semantic_paraphrase"),
        ("After a corporate notebook goes missing, what is the reporting deadline?", ["security_laptop"], "report within one hour", "semantic_paraphrase"),
        ("How long does a critical supplier have to disclose a material incident?", ["legal_vendors"], "within 24 hours", "semantic_paraphrase"),
        ("What minimum number of characters must a Northstar password contain?", ["security_passwords"], "14 characters", "semantic_paraphrase"),
        ("Who signs off on the contractual terms of a supplier?", ["legal_vendors"], "Legal", "semantic_paraphrase"),
        ("Which team owns technical risk for a vendor?", ["legal_vendors"], "Security", "entity_or_acronym"),
        ("What agreement is required for a vendor handling personal data?", ["legal_vendors"], "data-processing agreement", "entity_or_acronym"),
    ]
    for question, docs, answer, category in additional_questions:
        rows.append({
            "id": f"q{index:03d}",
            "question": question,
            "relevant_document_ids": docs,
            "relevant_chunk_ids": [],
            "expected_answer": answer,
            "question_type": category,
            "category": category,
        })
        index += 1
    EVAL.mkdir(parents=True, exist_ok=True)
    (EVAL / "retrieval_questions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    agent_questions = [
        {"question": "What is the remote work policy?", "expected_tools": ["document_search"]},
        {"question": "How many employees work in France?", "expected_tools": ["sql_query"]},
        {"question": "What percentage of employees work in France?", "expected_tools": ["sql_query", "calculator"]},
        {"question": "What is the password expiration policy?", "expected_tools": ["document_search"]},
        {"question": "What are the average travel expenses?", "expected_tools": ["sql_query"]},
    ]
    (EVAL / "agent_questions.json").write_text(json.dumps(agent_questions, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reproducible synthetic enterprise corpus.")
    parser.add_argument("--generated-count", type=int, default=600, help="Number of scalable synthetic documents to add.")
    args = parser.parse_args()
    if args.generated_count < 0:
        raise SystemExit("--generated-count must be zero or greater")
    random.seed(42)
    records = write_documents(args.generated_count)
    write_evaluation(records)
    print(f"Created {len(records)} documents ({args.generated_count} scalable synthetic documents) and 70 evaluation questions.")


if __name__ == "__main__":
    main()
