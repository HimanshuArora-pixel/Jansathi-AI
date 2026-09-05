---
language:
- en
- hi
license: apache-2.0
tags:
- legal
- text-classification
- bert
- inlegalbert
- jansaathi
library_name: transformers
pipeline_tag: text-classification
widget:
- text: "My landlord has refused to return my security deposit for 3 months"
- text: "How to file an RTI to get municipal road repair expenditure"
- text: "Received a cheque bounce notice under section 138"
---

# JanSaathi InLegalBERT Intent Classifier

Custom fine-tuned legal intent classification model for **JanSaathi** (AI for Civic & Legal Empowerment).

## Classes
- `Cheque_Bounce`
- `Civic_Scheme_Info`
- `Consumer_Dispute`
- `Criminal_FIR`
- `Cybercrime`
- `Legal_Notice_Contract`
- `RERA_RealEstate`
- `RTI`
- `Tenant_Landlord`
- `Workplace_Labour`
