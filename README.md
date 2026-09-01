# Pet First Aid Assistant

Pet First Aid Assistant is a safety-focused Retrieval-Augmented Generation
(RAG) application that provides source-grounded first-aid guidance for dogs
and cats.

Users can describe symptoms in their own words or select a predefined emergency.
The application retrieves relevant information from authoritative veterinary
sources and generates a structured response containing:

- an urgency assessment
- immediate first-aid steps
- actions to avoid
- safe transportation guidance
- source references
- a recommendation to contact a veterinarian

## Important Safety Notice

This application does not diagnose medical conditions and does not replace a
veterinarian.

It provides general first-aid information intended to support pet owners while
they contact or travel to a veterinary clinic. Any first aid provided to an
animal should be followed by appropriate veterinary care.

If an animal is unconscious, not breathing, having difficulty breathing,
bleeding severely, experiencing prolonged or repeated seizures, showing signs
of poisoning, or has suffered major trauma, contact an emergency veterinary
clinic immediately.

## Initial Scope

The first version supports:

- dogs
- cats

The initial emergency categories include:

- breathing difficulty
- choking
- unconsciousness
- cardiopulmonary arrest
- severe bleeding
- poisoning
- heatstroke
- seizures
- vehicle accidents and major trauma
- suspected fractures
- burns
- bite wounds
- persistent vomiting or diarrhea
- swollen or painful abdomen
- eye injuries

Other species and non-emergency healthcare advice are outside the initial
project scope.

## Planned Architecture

The application will contain:

1. A curated catalogue of authoritative veterinary sources
2. A data-ingestion and document-cleaning pipeline
3. Document chunking and metadata extraction
4. Keyword, vector, and hybrid retrieval
5. Retrieval evaluation using a ground-truth dataset
6. A safety-focused RAG generation pipeline
7. LLM answer and safety evaluation
8. A FastAPI application and web interface
9. User feedback collection and monitoring
10. Docker-based local deployment

## Authoritative Sources

The initial knowledge base will use information from organizations including:

- American Veterinary Medical Association
- Merck Veterinary Manual
- VCA Animal Hospitals
- American Red Cross

Every generated answer must include the sources used to produce it.

## Project Status

The project is currently under active development as the final project for the
DataTalks.Club LLM Zoomcamp 2026 cohort.

## Technology Stack

The planned technology stack includes:

- Python 3.12
- FastAPI
- OpenAI API
- keyword and vector retrieval
- PostgreSQL or SQLite for application data
- pandas for evaluation
- pytest
- Docker and Docker Compose

The final selection of retrieval and monitoring technologies will be documented
as the project develops.

## License and Source Attribution

The application uses external veterinary materials as referenced source
material. Ownership of those materials remains with their respective
publishers. The project will store source attribution and links alongside
retrieved content.