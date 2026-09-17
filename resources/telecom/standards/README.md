# Telecom standards model inputs

The runtime registry is built from normalized, versioned model artifacts shipped with the application under `resources/telecom/standards/normalized/`.

The telecom MVP intentionally composes three standards families rather than asking the LLM to choose a standard:

- **TM Forum** — Information Framework / GB922 and TMF654 Prepay Balance Management for CSP customer, account, product and prepaid-balance concepts.
- **3GPP** — charging specifications such as TS 32.240, TS 32.296, TS 32.297 and TS 32.298 for charging, online charging and charging-record concepts.
- **MEF** — MEF 125 / MEF 6.3 for Subscriber Ethernet and MEF 139 for Internet Access service/product concepts.

The runtime registry derives applicable standards from entity provenance. The LLM can request concepts, but it cannot assert which standard is authoritative and cannot create runtime entities or relationships.

## Artifact status

The checked-in JSON files are **INGENII normalized MVP subsets with source provenance**; they are not verbatim copies of the standards. For production certification, pin the exact TM Forum, 3GPP and MEF releases required by the deployment and ingest the approved machine-readable source artifacts with the provided ingestion tooling.
